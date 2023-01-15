module test_backslash
using Test
using ExtendableSparse
using ForwardDiff
using MultiFloats
using LinearAlgebra
f64(x::ForwardDiff.Dual{T}) where {T} = Float64(ForwardDiff.value(x))
f64(x::Number) = Float64(x)
const Dual64 = ForwardDiff.Dual{Float64, Float64, 1}
Base.zero(TT::Type{ForwardDiff.Dual{T, T, N}}) where {T, N} = TT(zero(T))

function test_bslash(T, k, l, m)
    A = fdrand(T, k, l, m; rand = () -> one(T), matrixtype = ExtendableSparseMatrix)
    x = ones(T, size(A, 1))
    b = A * x
    norm(f64.(A \ b - x))
end

function test_bslash(T)
    @info "$T:"
    atol = 10.0 * sqrt(f64(eps(T)))
    @test isapprox(test_bslash(T, 100, 1, 1), 0.0; atol)
    @test isapprox(test_bslash(T, 20, 20, 1), 0.0; atol)
    @test isapprox(test_bslash(T, 10, 10, 10), 0.0; atol)
end

test_bslash(Float32)
test_bslash(Float64)
test_bslash(Float64x1)
test_bslash(Float64x2)
test_bslash(ForwardDiff.Dual{Float64, Float64, 1})
test_bslash(ForwardDiff.Dual{Float64, Float64, 2})

end
