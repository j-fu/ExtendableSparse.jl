module ExperimentalXParallel

using ExtendableSparse, SparseArrays, ExtendableSparse.Experimental
using BenchmarkTools
using ExtendableGrids
#using MKLSparse
#using SparseMatricesCSR
using Test

using ExtendableSparse, ExtendableGrids, Metis
using LinearAlgebra
using BenchmarkTools
using Test
using OhMyThreads: @tasks
using RecursiveFactorization

function testgrid(N; dim=3)
    X = range(0, 1; length=N^(1.0 / dim) |> ceil |> Int)
    simplexgrid((X for i = 1:dim)...)
end

function coordmatrix!(C, coord, cellnodes, k)
    spacedim=size(coord,1)
    celldim=size(cellnodes,1)
    @inbounds for jj = 1:celldim
        C[1, jj] = 1
        @inbounds for ii = 1:spacedim
            C[ii + 1, jj] = coord[ii, cellnodes[jj, k]]
        end
    end
end

function gradient!(G, C, factdim, I, ipiv)
    clu = RecursiveFactorization.lu!(C, ipiv, Val(true), Val(false))
    ldiv!(G, clu, I)
    abs(det(clu)) / factdim
end

function scalpro(G, dim, jl, il)
    s = 0.0
    @inbounds @simd for k = 1:dim
        s += G[jl, k + 1] * G[il, k + 1]
    end
    return s
end

function stiffness!(S, dim, G)
    @inbounds for il = 1:(dim + 1)
        S[il, il] = scalpro(G, dim, il, il)
        @inbounds for jl = (il + 1):(dim + 1)
            S[il, jl] = scalpro(G, dim, jl, il)
            S[jl, il] = S[il, jl]
        end
    end
    return S
end

function testassemble!(A_h, grid)
    coord = grid[Coordinates]
    cellnodes = grid[CellNodes]
    ncells = num_cells(grid)
    dim = size(coord, 1)
    lnodes = dim + 1
    factdim::Float64 = factorial(dim)
    S = zeros(lnodes, lnodes) # local stiffness matrix
    C = zeros(lnodes, lnodes)  # local coordinate matrix
    G = zeros(lnodes, lnodes) # shape function gradients
    ipiv = zeros(Int, lnodes)
    I = Matrix(Diagonal(ones(lnodes)))
    ncells = size(cellnodes, 2)
    for icell = 1:ncells
        coordmatrix!(C, coord, cellnodes, icell)
        vol = gradient!(G, C, factdim, I, ipiv)
        stiffness!(S, dim, G)
        for il = 1:lnodes
            i = cellnodes[il, icell]
            rawupdateindex!(A_h, +, 0.1 * vol / (dim + 1), i, i)
            for jl = 1:lnodes
                j = cellnodes[jl, icell]
                rawupdateindex!(A_h, +, vol * (S[il, jl]), i, j)
            end
        end
    end
    flush!(A_h)
end

function testassemble_parallel!(A_h, grid)
    coord = grid[Coordinates]
    cellnodes = grid[CellNodes]
    ncells = num_cells(grid)
    dim = size(coord, 1)
    lnodes = dim + 1
    npart = num_partitions(grid)
    factdim::Float64 = factorial(dim)
    SS = [zeros(lnodes, lnodes) for i = 1:npart] # local stiffness matrix
    CC = [zeros(lnodes, lnodes) for i = 1:npart] # local coordinate matrix
    GG = [zeros(lnodes, lnodes) for i = 1:npart] # shape function gradients
    IP = [zeros(Int, lnodes) for i = 1:npart] # shape function gradients
    I = Matrix(Diagonal(ones(lnodes)))
    ncells = size(cellnodes, 2)
    for color in pcolors(grid)
        @tasks for part in pcolor_partitions(grid, color)
            C = CC[part]
            S = SS[part]
            G = GG[part]
            ipiv = IP[part]
            for icell in partition_cells(grid, part)
                coordmatrix!(C, coord, cellnodes, icell)
                vol = gradient!(G, C, factdim, I, ipiv)
                stiffness!(S, dim, G)
                for il = 1:lnodes
                    i = cellnodes[il, icell]
                    rawupdateindex!(A_h, +, 0.1 * vol / (dim + 1), i, i, part)
                    for jl = 1:lnodes
                        j = cellnodes[jl, icell]
                        rawupdateindex!(A_h, +, vol * (S[il, jl]), i, j, part)
                    end
                end
            end
        end
    end
    flush!(A_h)
end

function test_correctness_build_seq(N, Tm::Type{<:AbstractSparseMatrix}; dim=3)
    grid = testgrid(N; dim)
    nnodes = num_nodes(grid)
    A0 = ExtendableSparseMatrix{Float64,Int}(nnodes, nnodes)
    A = Tm{Float64,Int}(nnodes, nnodes)
    testassemble!(A0, grid)
    testassemble!(A, grid)
    @test sparse(A0) ≈ sparse(A)
end

function speedup_build_seq(N, Tm::Type{<:AbstractSparseMatrix}; dim=3)
    grid = testgrid(N; dim)
    nnodes = num_nodes(grid)
    A0 = ExtendableSparseMatrix{Float64,Int}(nnodes, nnodes)
    A = Tm{Float64,Int}(nnodes, nnodes)
    tbase = @belapsed testassemble!($A0, $grid) seconds = 1 setup = (reset!($A0))
    tx = @belapsed testassemble!($A, $grid) seconds = 1 setup = (reset!($A))
    tbase / tx
end

function test_correctness_update(N,
                                 Tm::Type{<:AbstractSparseMatrix};
                                 Tp::Type{<:AbstractPartitioningAlgorithm}=PlainMetisPartitioning,
                                 allnp=[10, 15, 20],
                                 dim=3)
    grid = testgrid(N; dim)
    nnodes = num_nodes(grid)
    A = Tm{Float64,Int}(nnodes, nnodes, 1)

    # Assembele without partitioning
    # this gives the "base truth" to compare with
    testassemble_parallel!(A, grid)

    # Save the nonzeros 
    nz = sort(copy(nonzeros(A)))
    for np in allnp
        # Reset the nonzeros, keeping the structure intact
        nonzeros(A) .= 0
        # Parallel assembly whith np threads
        pgrid = partition(grid, Tp(; npart=np))
        reset!(A, np)
        @show num_partitions_per_color(pgrid)
        testassemble_parallel!(A, pgrid)
        @test sort(nonzeros(A)) ≈ nz
    end
end

"""
    test_correctness_build(N)

Test correctness of parallel assembly on NxN grid  during 
build phase, assuming that no structure has been assembled.
"""
function test_correctness_build(N,
                                Tm::Type{<:AbstractSparseMatrix};
                                Tp::Type{<:AbstractPartitioningAlgorithm}=PlainMetisPartitioning,
                                allnp=[10, 15, 20],
                                dim=3)
    grid = testgrid(N; dim)
    nnodes = num_nodes(grid)
    # Get the "ground truth"
    A0 = ExtendableSparseMatrix{Float64,Int}(nnodes, nnodes)
    testassemble!(A0, grid)
    nz = sort(copy(nonzeros(A0)))
    for np in allnp
        # Make a new matrix and assemble parallel.
        # this should result in the same nonzeros
        pgrid = partition(grid, Tp(; npart=np))
        A = Tm(nnodes, nnodes, num_partitions(pgrid))
        @show num_partitions_per_color(pgrid)
        @test check_partitioning(pgrid)
        testassemble_parallel!(A, pgrid)
        @test sort(nonzeros(A)) ≈ nz
    end
end

function test_correctness_mul(N,
                              Tm::Type{<:AbstractSparseMatrix};
                              Tp::Type{<:AbstractPartitioningAlgorithm}=PlainMetisPartitioning,
                              allnp=[10, 15, 20],
                              dim=3)
    grid = testgrid(N; dim)
    nnodes = num_nodes(grid)
    # Get the "ground truth"
    A0 = ExtendableSparseMatrix{Float64,Int}(nnodes, nnodes)
    testassemble!(A0, grid)
    b = rand(nnodes)
    A0b = A0 * b
    for np in allnp
        pgrid = partition(grid, Tp(; npart=np))
        @test check_partitioning(pgrid)
        A = Tm(nnodes, nnodes, num_partitions(pgrid))
        ExtendableSparse.Experimental.partitioning!(A, pgrid[PColorPartitions],
                                                    pgrid[PartitionNodes])
        testassemble_parallel!(A, pgrid)
        invp = invperm(pgrid[NodePermutation])
        diff = norm(A0b[invp] - A * b[invp], Inf)
        @show diff
        @test diff < sqrt(eps())
    end
end

function speedup_update(N,
                        Tm::Type{<:AbstractSparseMatrix};
                        Tp::Type{<:AbstractPartitioningAlgorithm}=PlainMetisPartitioning,
                        allnp=[10, 15, 20],
                        dim=3)
    grid = testgrid(N; dim)
    nnodes = num_nodes(grid)
    # Get the "ground truth"
    A0 = ExtendableSparseMatrix{Float64,Int}(nnodes, nnodes)
    testassemble!(A0, grid)
    nz = copy(nonzeros(A0)) |> sort
    # Get the base timing
    # During setup, set matrix entries to zero while keeping  the structure 
    t0 = @belapsed testassemble!($A0, $grid) seconds = 1 setup = (nonzeros($A0) .= 0)
    result = []
    A = Tm(nnodes, nnodes, 1)
    for np in allnp
        # Get the parallel timing
        # During setup, set matrix entries to zero while keeping  the structure
        pgrid = partition(grid, Tp(; npart=np))
        @show num_partitions_per_color(pgrid)
        reset!(A, num_partitions(pgrid))
        testassemble_parallel!(A, pgrid)
        t = @belapsed testassemble_parallel!($A, $pgrid) seconds = 1 setup = (nonzeros($A) .= 0)
        @assert sort(nonzeros(A)) ≈ nz
        push!(result, (np, round(t0 / t; digits=2)))
    end
    result
end

function speedup_build(N,
                       Tm::Type{<:AbstractSparseMatrix};
                       Tp::Type{<:AbstractPartitioningAlgorithm}=PlainMetisPartitioning,
                       allnp=[10, 15, 20],
                       dim=3)
    grid = testgrid(N; dim)
    nnodes = num_nodes(grid)
    # Get the "ground truth"
    A0 = ExtendableSparseMatrix{Float64,Int}(nnodes, nnodes)
    testassemble!(A0, grid)
    nz = nonzeros(A0)
    reset!(A0)
    testassemble!(A0, grid)
    @assert nonzeros(A0) ≈ (nz)
    nz = sort(nz)

    # Get the base timing
    # During setup, reset matrix to empty state.
    t0 = @belapsed testassemble!($A0, $grid) seconds = 1 setup = (reset!($A0))

    result = []
    A = Tm(nnodes, nnodes, 1)
    for np in allnp
        # Get the parallel timing
        # During setup, reset matrix to empty state.
        pgrid = partition(grid, Tp(; npart=np))
        reset!(A, num_partitions(pgrid))
        @show num_partitions_per_color(pgrid)
        t = @belapsed testassemble_parallel!($A, $pgrid) seconds = 1 setup = (reset!($A,
                                                                                     num_partitions($pgrid)))
        @assert sort(nonzeros(A)) ≈ nz
        push!(result, (np, round(t0 / t; digits=2)))
    end
    result
end

function speedup_mul(N,
                     Tm::Type{<:AbstractSparseMatrix};
                     Tp::Type{<:AbstractPartitioningAlgorithm}=PlainMetisPartitioning,
                     allnp=[10, 15, 20],
                     dim=3)
    grid = testgrid(N; dim)
    nnodes = num_nodes(grid)
    # Get the "ground truth"
    A0 = ExtendableSparseMatrix{Float64,Int}(nnodes, nnodes)
    testassemble!(A0, grid)
    b = rand(nnodes)
    t0 = @belapsed $A0 * $b seconds = 1
    A0b = A0 * b
    result = []
    A = Tm(nnodes, nnodes, 1)
    for np in allnp
        pgrid = partition(grid, Tp(; npart=np))
        @show num_partitions_per_color(pgrid)
        reset!(A, num_partitions(pgrid))
        testassemble_parallel!(A, pgrid)
        flush!(A)
        ExtendableSparse.Experimental.partitioning!(A, pgrid[PColorPartitions],
                                                    pgrid[PartitionNodes])
        t = @belapsed $A * $b seconds = 1
        invp = invperm(pgrid[NodePermutation])
        @assert A0b[invp] ≈ A * b[invp]
        push!(result, (np, round(t0 / t; digits=2)))
    end
    result
end

#=
function mymul(A::SparseMatrixCSR,v::AbstractVector)
    y=copy(v)
    A.n == size(v, 1) || throw(DimensionMismatch())
    A.m == size(y, 1) || throw(DimensionMismatch())
    @tasks for row = 1:size(y, 1)
        y[row]=0.0
        @inbounds for nz in nzrange(A,row)
            col = A.colval[nz]
            y[row] += A.nzval[nz]*v[col]
        end
    end
    return y
end

function speedup_csrmul(N; dim=3)
    grid=testgrid(N;dim)
    nnodes=num_nodes(grid)
    # Get the "ground truth"
    A0=ExtendableSparseMatrix{Float64,Int}(nnodes,nnodes)
    t00=@belapsed testassemble!($A0,$grid) seconds=1 setup=(reset!($A0))

    reset!(A0)
    testassemble!(A0,grid)
    b=rand(nnodes)
    t0=@belapsed $A0*$b seconds=1
    A0b=A0*b

    t0x=@belapsed  A0x=sparse(transpose(sparse($A0)))

    A0x=sparse(transpose(sparse(A0)))

    tx=@belapsed A=SparseMatrixCSR{1}(transpose($A0x))

    A=SparseMatrixCSR{1}(transpose(sparse(A0x)))
    t1=@belapsed $A*$b seconds=1

    t2=@belapsed mymul($A, $b) seconds=1

    @info t00,t0,t0x, tx,t1, t2

    @assert A0b≈A*b
    t0/t1
end

=#

end
