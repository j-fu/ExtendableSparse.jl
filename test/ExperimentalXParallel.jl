module ExperimentalXParallel

using ExtendableSparse,SparseArrays, ExtendableSparse.Experimental
using BenchmarkTools
using ExtendableGrids
#using MKLSparse
using SparseMatricesCSR
using Test
using OhMyThreads

include("test_parallel.jl")

function test_correctness_update(N,Tm::Type{<:AbstractSparseMatrix}; dim=3)
    grid=testgrid(N;dim)
    nnodes=num_nodes(grid)
    A=Tm{Float64,Int}(nnodes,nnodes,1)
    allnp=[10,15,20]

    # Assembele without partitioning
    # this gives the "base truth" to compare with
    testassemble_parallel!(A,grid)

    # Save the nonzeros 
    nz=copy(nonzeros(A))
    for np in allnp
        # Reset the nonzeros, keeping the structure intact
        nonzeros(A).=0
        # Parallel assembly whith np threads
        pgrid=partition(grid,PlainMetisPartitioning(npart=np))
        @show num_partitions_per_color(pgrid)
        testassemble_parallel!(A,pgrid)
        @test nonzeros(A)≈nz
    end
end

"""
    test_correctness_build(N)

Test correctness of parallel assembly on NxN grid  during 
build phase, assuming that no structure has been assembled.
"""
function test_correctness_build(N,Tm::Type{<:AbstractSparseMatrix}; allnp=[10,15,20], dim=3)
    grid=testgrid(N;dim)
    nnodes=num_nodes(grid)
    # Get the "ground truth"
    A0=ExtendableSparseMatrix{Float64,Int}(nnodes,nnodes)
    testassemble!(A0,grid)
    nz=copy(nonzeros(A0))
    for np in allnp
        # Make a new matrix and assemble parallel.
        # this should result in the same nonzeros
        pgrid=partition(grid,PlainMetisPartitioning(npart=np))
        A=Tm(nnodes,nnodes, num_partitions(pgrid))
        @show num_partitions_per_color(pgrid)
        @test checkpartitioning(pgrid)
        testassemble_parallel!(A,pgrid)
        @test nonzeros(A) ≈ nz
    end
end

function test_correctness_mul(N,Tm::Type{<:AbstractSparseMatrix}; allnp=[10,15,20], dim=3)
    grid=testgrid(N;dim)
    nnodes=num_nodes(grid)
    # Get the "ground truth"
    A0=ExtendableSparseMatrix{Float64,Int}(nnodes,nnodes)
    testassemble!(A0,grid)
    for np in allnp
        pgrid=partition(grid,PlainMetisPartitioning(npart=np))
        A=Tm(nnodes,nnodes, num_partitions(pgrid))
        testassemble_parallel!(A,pgrid)
        flush!(A)
        partcolors!(A,partition_pcolors(pgrid))
        b=rand(nnodes)
        @test A*b ≈ A0*b
    end    
end

function speedup_update(N,Tm::Type{<:AbstractSparseMatrix}; allnp=[10,15,20], dim=3)
    grid=testgrid(N;dim)
    nnodes=num_nodes(grid)
    # Get the "ground truth"
    A0=ExtendableSparseMatrix{Float64,Int}(nnodes,nnodes)
    testassemble!(A0,grid)
    nz=copy(nonzeros(A0))
    # Get the base timing
    # During setup, set matrix entries to zero while keeping  the structure 
    t0=@belapsed testassemble!($A0,$grid) seconds=1 setup=(nonzeros($A0).=0)
    result=[]
    A=Tm(nnodes,nnodes,1)
    for np in allnp
        # Get the parallel timing
        # During setup, set matrix entries to zero while keeping  the structure
        pgrid=partition(grid,PlainMetisPartitioning(npart=np))
        @show num_partitions_per_color(pgrid)
        reset!(A,num_partitions(pgrid))
        testassemble_parallel!(A,pgrid)
        t=@belapsed testassemble_parallel!($A,$pgrid) seconds=1 setup=(nonzeros($A).=0)
        @assert nonzeros(A)≈nz
        push!(result,(np,round(t0/t,digits=2)))
    end
    result
end

function speedup_build(N,Tm::Type{<:AbstractSparseMatrix}; allnp=[10,15,20], dim=3)
    grid=testgrid(N;dim)
    nnodes=num_nodes(grid)
    # Get the "ground truth"
    A0=ExtendableSparseMatrix{Float64,Int}(nnodes,nnodes)
    testassemble!(A0,grid)
    nz=copy(nonzeros(A0))
    reset!(A0)
    testassemble!(A0,grid)
    @assert nonzeros(A0)≈(nz)

    # Get the base timing
    # During setup, reset matrix to empty state.
    t0=@belapsed testassemble!($A0,$grid) seconds=1 setup=(reset!($A0))
    
    result=[]
    A=Tm(nnodes,nnodes,1)
    for np in allnp
        # Get the parallel timing
        # During setup, reset matrix to empty state.
        pgrid=partition(grid,PlainMetisPartitioning(npart=np))
        reset!(A,num_partitions(pgrid))
        @show num_partitions_per_color(pgrid)
        t=@belapsed testassemble_parallel!($A,$pgrid) seconds=1 setup=(reset!($A,num_partitions($pgrid)))
        @assert nonzeros(A)≈nz
        push!(result,(np,round(t0/t,digits=2)))
    end
    result
end

function speedup_mul(N,Tm::Type{<:AbstractSparseMatrix}; allnp=[10,15,20], dim=3)
    grid=testgrid(N;dim)
    nnodes=num_nodes(grid)
    # Get the "ground truth"
    A0=ExtendableSparseMatrix{Float64,Int}(nnodes,nnodes)
    testassemble!(A0,grid)
    b=rand(nnodes)
    t0=@belapsed $A0*$b seconds=1
    A0b=A0*b
    result=[]
    A=Tm(nnodes,nnodes,1)
    for np in allnp
        pgrid=partition(grid,PlainMetisPartitioning(npart=np))
        @show num_partitions_per_color(pgrid)
        reset!(A,num_partitions(pgrid))
        testassemble_parallel!(A,pgrid)
        flush!(A)
        partcolors!(A,partition_pcolors(pgrid))

        t=@belapsed $A*$b seconds=1
        @assert A0b≈A*b
        push!(result,(np,round(t0/t,digits=2)))
    end
    result
end


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


end

