module ExperimentalDict

using ExtendableSparse,SparseArrays, ExtendableSparse.Experimental
using BenchmarkTools
using Test


function ExtendableSparse.reset!(A::ExtendableSparseMatrix)
    A.cscmatrix=spzeros(size(A)...)
    A.lnkmatrix=nothing
end


function test_correctness_build(N)
    X=1:N
    Y=1:N
    A0=ExtendableSparseMatrix{Float64,Int}(N^2,N^2)
    A=ExtendableSparseMatrixDict{Float64,Int}(N^2,N^2)
    partassemble!(A0,X,Y)
    partassemble!(A,X,Y)
    @test sparse(A0)≈sparse(A)
end

function speed_build(N)
    X=1:N
    Y=1:N
    A0=ExtendableSparseMatrix{Float64,Int}(N^2,N^2)
    A=ExtendableSparseMatrixDict{Float64,Int}(N^2,N^2)

    tlnk= @belapsed partassemble!($A0,$X,$Y) seconds=1 setup=(reset!($A0))
    tdict= @belapsed partassemble!($A,$X,$Y) seconds=1 setup=(reset!($A))
    tdict/tlnk
end

end
