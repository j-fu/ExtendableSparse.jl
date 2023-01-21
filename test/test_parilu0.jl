module test_parilu0
using Test
using ExtendableSparse
using IterativeSolvers
using LinearAlgebra

function test(n)
    A = ExtendableSparseMatrix(n, n)
    sprand_sdd!(A)
    flush!(A)
    A = A.cscmatrix
    b = A * ones(n)
    P_par = ParallelILU0Preconditioner(A)
    A_reord, b_reord = reorderlinsys(A, b, P_par.coloring)
    P_ser = ILU0Preconditioner(A_reord)
    sol_ser, hist_ser = gmres(A_reord, b_reord; Pl = P_ser, log = true)
    sol_par, hist_par = gmres(A_reord, b_reord; Pl = P_par, log = true)
    sol_ser == sol_par && hist_ser.iters == hist_par.iters
end

@test test(10)
@test test(100)
@test test(1000)
end
