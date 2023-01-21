module test_default_cholesky
using Test
using ExtendableSparse
using SparseArrays
using LinearAlgebra

include("test_lu.jl")

@test test_lu1(Float64, 10, 10, 10, lufac = CholeskyFactorization())
@test test_lu1(Float64, 25, 40, 1, lufac = CholeskyFactorization())
@test test_lu1(Float64, 1000, 1, 1, lufac = CholeskyFactorization())

@test test_lu2(Float64, 10, 10, 10, lufac = CholeskyFactorization())
@test test_lu2(Float64, 25, 40, 1, lufac = CholeskyFactorization())
@test test_lu2(Float64, 1000, 1, 1, lufac = CholeskyFactorization())
end
