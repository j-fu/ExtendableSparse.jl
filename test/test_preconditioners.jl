module test_preconditioners
using Test
using ExtendableSparse
using AlgebraicMultigrid
using IncompleteLU
using IterativeSolvers
using LinearAlgebra

function test_precon(Precon,k,l,m;maxiter=10000,symmetric=true)
    A=fdrand(k,l,m;matrixtype=ExtendableSparseMatrix,symmetric)
    b=ones(size(A,2))
    exact=A\b
    Pl=Precon(A)
    it,hist=simple(A,b,Pl=Pl,maxiter=maxiter,reltol=1.0e-10,log=true)
    r=hist[:resnorm]
    nr=length(r)
    tail=min(100,length(r)÷2)
    all(x-> x<1,r[end-tail:end]./r[end-tail-1:end-1]),norm(it-exact)
end


function test_precon2(precon,k,l,m;maxiter=10000,symmetric=true)
    A=fdrand(k,l,m;matrixtype=ExtendableSparseMatrix, symmetric)
    b=ones(size(A,2))
    exact=A\b
    factorize!(precon,A)
    it,hist=simple(A,b,Pl=precon,maxiter=maxiter,reltol=1.0e-10,log=true)
    r=hist[:resnorm]
    nr=length(r)
    tail=min(100,length(r)÷2)
    all(x-> x<1,r[end-tail:end]./r[end-tail-1:end-1]),norm(it-exact)
end



@test   all(isapprox.(test_precon(ILU0Preconditioner,20,20,20),           (true, 1.3535160424212675e-5), rtol=1.0e-3))
@test   all(isapprox.(test_precon(JacobiPreconditioner,20,20,20),         (true, 2.0406032775945658e-5), rtol=1.0e-3))
@test   all(isapprox.(test_precon(ParallelJacobiPreconditioner,20,20,20), (true, 2.0406032775945658e-5), rtol=1.0e-3))
@test   all(isapprox.(test_precon(ILUTPreconditioner,20,20,20),           (true, 1.2719511868322086e-5), rtol=1.0e-3))
@test   all(isapprox.(test_precon(AMGPreconditioner,20,20,20),            (true, 6.863753664354144e-7), rtol=1.0e-2))

@test   all(isapprox.(test_precon2(ILU0Preconditioner(),20,20,20),           (true, 1.3535160424212675e-5), rtol=1.0e-3))
@test   all(isapprox.(test_precon2(JacobiPreconditioner(),20,20,20),         (true, 2.0406032775945658e-5), rtol=1.0e-3))
@test   all(isapprox.(test_precon2(ParallelJacobiPreconditioner(),20,20,20), (true, 2.0406032775945658e-5), rtol=1.0e-3))
@test   all(isapprox.(test_precon2(ILUTPreconditioner(),20,20,20),           (true, 1.2719511868322086e-5), rtol=1.0e-3))
@test   all(isapprox.(test_precon2(AMGPreconditioner(),20,20,20),            (true, 6.863753664354144e-7), rtol=1.0e-2))

end
