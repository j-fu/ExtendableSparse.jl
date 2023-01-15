module test_linearsolve

using Test
using ExtendableSparse
using SparseArrays
using LinearAlgebra
using LinearSolve
using ForwardDiff
using MultiFloats

f64(x::ForwardDiff.Dual{T}) where {T} = Float64(ForwardDiff.value(x))
f64(x::Number) = Float64(x)
const Dual64 = ForwardDiff.Dual{Float64, Float64, 1}

function test_ls1(T, k, l, m; linsolver = SparspakFactorization())
    A = fdrand(T, k, l, m; rand = () -> 1, matrixtype = ExtendableSparseMatrix)
    b = rand(k * l * m)
    x0 = A \ b

    p = LinearProblem(A, T.(b))
    x1 = solve(p, linsolver)
    x0 ≈ x1
end

@test test_ls1(Float64, 10, 10, 10, linsolver = KLUFactorization())
@test test_ls1(Float64, 25, 40, 1, linsolver = KLUFactorization())
@test test_ls1(Float64, 1000, 1, 1, linsolver = KLUFactorization())

for T in [Float32, Float64, Float64x1, Float64x2, Dual64]
    println("$T:")
    @test test_ls1(T, 10, 10, 10, linsolver = SparspakFactorization())
    @test test_ls1(T, 25, 40, 1, linsolver = SparspakFactorization())
    @test test_ls1(T, 1000, 1, 1, linsolver = SparspakFactorization())
end

function test_ls2(T, k, l, m; linsolver = SparspakFactorization())
    A = fdrand(T, k, l, m; rand = () -> 1, matrixtype = ExtendableSparseMatrix)
    b = T.(rand(k * l * m))
    p = LinearProblem(A, b)
    x0 = solve(p, linsolver)
    cache = x0.cache
    x0 = A \ b
    for i = 4:(k * l * m - 3)
        A[i, i + 3] -= 1.0e-4
        A[i - 3, i] -= 1.0e-4
    end

    LinearSolve.set_A(cache, A)
    x1 = solve(p, linsolver)
    x1 = A \ b

    all(x0 .< x1)
end

@test test_ls1(Float64, 10, 10, 10, linsolver = KLUFactorization())
@test test_ls1(Float64, 25, 40, 1, linsolver = KLUFactorization())
@test test_ls1(Float64, 1000, 1, 1, linsolver = KLUFactorization())

for T in [Float32, Float64, Float64x1, Float64x2, Dual64]
    println("$T:")
    @test test_ls1(T, 10, 10, 10, linsolver = SparspakFactorization())
    @test test_ls1(T, 25, 40, 1, linsolver = SparspakFactorization())
    @test test_ls1(T, 1000, 1, 1, linsolver = SparspakFactorization())
end

@test test_ls2(Float64, 10, 10, 10, linsolver = KLUFactorization())
@test test_ls2(Float64, 25, 40, 1, linsolver = KLUFactorization())
@test test_ls2(Float64, 1000, 1, 1, linsolver = KLUFactorization())

for T in [Float32, Float64, Float64x1, Float64x2, Dual64]
    println("$T:")
    @test test_ls2(T, 10, 10, 10, linsolver = SparspakFactorization())
    @test test_ls2(T, 25, 40, 1, linsolver = SparspakFactorization())
    @test test_ls2(T, 1000, 1, 1, linsolver = SparspakFactorization())
end

end
