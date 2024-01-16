module test_preconditioners
using Test
using ExtendableSparse
using AlgebraicMultigrid
using AMGCLWrap
using IncompleteLU
using IterativeSolvers
using LinearAlgebra

function test_precon(Precon, k, l, m; maxiter = 10000, symmetric = true)
    A = fdrand(k, l, m; matrixtype = ExtendableSparseMatrix, symmetric)
    b = ones(size(A, 2))
    exact = A \ b
    Pl = Precon(A)
    it, hist = simple(A, b; Pl = Pl, maxiter = maxiter, reltol = 1.0e-10, log = true)
    r = hist[:resnorm]
    nr = length(r)
    tail = min(100, length(r) ÷ 2)
    all(x -> x < 1, r[(end - tail):end] ./ r[(end - tail - 1):(end - 1)]), norm(it - exact)
end

function test_precon2(precon, k, l, m; maxiter = 10000, symmetric = true)
    A = fdrand(k, l, m; matrixtype = ExtendableSparseMatrix, symmetric)
    b = ones(size(A, 2))
    exact = A \ b
    @show typeof(precon)
    factorize!(precon, A)
    @time it, hist = simple(A, b; Pl = precon, maxiter = maxiter, reltol = 1.0e-10, log = true)
    r = hist[:resnorm]
    nr = length(r)
    tail = min(100, length(r) ÷ 2)
    all(x -> x < 1, r[(end - tail):end] ./ r[(end - tail - 1):(end - 1)]), norm(it - exact)
end

@test all(test_precon(ILU0Preconditioner, 20, 20, 20) .≤ (true, 4e-5))
@test all(test_precon(ILUZeroPreconditioner, 20, 20, 20) .≤ (true, 4e-5))
@test all(test_precon(JacobiPreconditioner, 20, 20, 20) .≤ (true, 3e-4))
@test all(test_precon(ParallelJacobiPreconditioner, 20, 20, 20) .≤ (true, 3e-4))
@test all(test_precon(ILUTPreconditioner, 20, 20, 20) .≤ (true, 5e-5))
@test all(test_precon(AMGPreconditioner, 20, 20, 20) .≤ (true, 1e-5))
@test all(test_precon(AMGCL_AMGPreconditioner, 20, 20, 20) .≤ (true, 1e-5))
@test all(test_precon(AMGCL_RLXPreconditioner, 20, 20, 20) .≤ (true, 4e-5))

@test all(test_precon(ILU0Preconditioner, 20, 20, 20; symmetric = false) .≤ (true, 4e-5))
@test all(test_precon(ILUZeroPreconditioner, 20, 20, 20; symmetric = false) .≤ (true, 4e-5))
@test all(test_precon(JacobiPreconditioner, 20, 20, 20; symmetric = false) .≤ (true, 3e-4))
@test all(test_precon(ParallelJacobiPreconditioner, 20, 20, 20; symmetric = false) .≤
          (true, 3e-4))
@test all(test_precon(ILUTPreconditioner, 20, 20, 20; symmetric = false) .≤ (true, 5e-5))
#@test   all(test_precon(AMGPreconditioner,20,20,20,symmetric=false).≤            (true, 1e-5))
#@test   all(test_precon(AMGCL_AMGPreconditioner,20,20,20,symmetric=false).≤            (true, 1e-5))
#@test   all(test_precon(AMGCL_RLXPreconditioner,20,20,20,symmetric=false).≤            (true, 5e-5))

@test all(test_precon2(ILU0Preconditioner(), 20, 20, 20) .≤ (true, 4e-5))
@test all(test_precon2(ILUZeroPreconditioner(), 20, 20, 20) .≤ (true, 4e-5))
@test all(test_precon2(JacobiPreconditioner(), 20, 20, 20) .≤ (true, 3e-4))
@test all(test_precon2(ParallelJacobiPreconditioner(), 20, 20, 20) .≤ (true, 3e-4))
@test all(test_precon2(ILUTPreconditioner(), 20, 20, 20) .≤ (true, 5e-5))
@test all(test_precon2(AMGPreconditioner(), 20, 20, 20) .≤ (true, 1e-5))
@test all(test_precon2(AMGCL_AMGPreconditioner(), 20, 20, 20) .≤ (true, 1e-5))
@test all(test_precon2(AMGCL_RLXPreconditioner(), 20, 20, 20) .≤ (true, 4e-5))

end
