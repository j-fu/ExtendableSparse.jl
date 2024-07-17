module test_parallel

using ExtendableSparse, SparseArrays
# using ExtendableSparse.Experimental
using BenchmarkTools
using ExtendableGrids
#using MKLSparse
#using SparseMatricesCSR
using Test

using ExtendableSparse, ExtendableGrids, Metis
using LinearAlgebra
using BenchmarkTools
using Test

include("femtools.jl")

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
        pgrid = partition(grid, Tp(; npart=np), nodes=true, keep_nodepermutation=true)
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
        pgrid = partition(grid, Tp(; npart=np), nodes=true, keep_nodepermutation=true)
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
        pgrid = partition(grid, Tp(; npart=np), nodes=true, keep_nodepermutation=true)
        @test check_partitioning(pgrid)
        A = Tm(nnodes, nnodes, num_partitions(pgrid))
        partitioning!(A, pgrid[PColorPartitions],
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
        pgrid = partition(grid, Tp(; npart=np), nodes=true, keep_nodepermutation=true)
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
        pgrid = partition(grid, Tp(; npart=np), nodes=true, keep_nodepermutation=true)
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
        pgrid = partition(grid, Tp(; npart=np), nodes=true, keep_nodepermutation=true)
        @show num_partitions_per_color(pgrid)
        reset!(A, num_partitions(pgrid))
        testassemble_parallel!(A, pgrid)
        flush!(A)
        partitioning!(A, pgrid[PColorPartitions],
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
