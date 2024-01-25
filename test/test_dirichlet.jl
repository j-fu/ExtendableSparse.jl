module test_dirichlet
using Test
using ExtendableSparse
using SparseArrays
using LinearAlgebra

function tdirichlet(A)
    n=size(A,1)
    for i=1:10:n
        A[i,i]=1.0e30
    end
    f=ones(n)
    u=A\f
    diri=mark_dirichlet(A)
    fD=f.* (1 .-diri)
    AD=eliminate_dirichlet(A,diri)
    uD=AD\fD
    norm(uD-u,Inf) < 1.0e-20
end

@test tdirichlet(fdrand(1000; matrixtype = SparseMatrixCSC))
@test tdirichlet(fdrand(1000; matrixtype = ExtendableSparseMatrix))

@test tdirichlet(fdrand(20,20; matrixtype = SparseMatrixCSC))
@test tdirichlet(fdrand(20,20; matrixtype = ExtendableSparseMatrix))

@test tdirichlet(fdrand(10,10,10, matrixtype = SparseMatrixCSC))
@test tdirichlet(fdrand(10,10,10; matrixtype = ExtendableSparseMatrix))

end
