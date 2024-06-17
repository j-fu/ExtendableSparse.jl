module ExperimentalScalar
using ExtendableSparse,SparseArrays, ExtendableSparse.Experimental
using BenchmarkTools
using Test

include("test_parallel.jl")

function test_correctness_build(N,Tm::Type{<:AbstractSparseMatrix}; dim=3)
    grid=testgrid(N;dim)
    nnodes=num_nodes(grid)
    A0=ExtendableSparseMatrix{Float64,Int}(nnodes,nnodes)
    A=Tm{Float64,Int}(nnodes,nnodes)
    testassemble!(A0,grid)
    testassemble!(A,grid)
    @test sparse(A0)≈sparse(A)
end

function speed_build(N,Tm::Type{<:AbstractSparseMatrix}; dim=3)
    grid=testgrid(N;dim)
    nnodes=num_nodes(grid)
    A0=ExtendableSparseMatrix{Float64,Int}(nnodes,nnodes)
    A=Tm{Float64,Int}(nnodes,nnodes)
    tbase= @belapsed testassemble!($A0,$grid) seconds=1 setup=(reset!($A0))
    tx= @belapsed testassemble!($A,$grid) seconds=1 setup=(reset!($A))
    tbase/tx
end

end
