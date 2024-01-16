module test_copymethods
using Test
using ExtendableSparse
using SparseArrays
using Random
using MultiFloats
using ForwardDiff

const Dual64 = ForwardDiff.Dual{Float64, Float64, 1}
function Random.rand(rng::AbstractRNG,
                     ::Random.SamplerType{ForwardDiff.Dual{T, V, N}}) where {T, V, N}
    ForwardDiff.Dual{T, V, N}(rand(rng, T))
end

function test(T)
    Xcsc = sprand(T, 10_000, 10_000, 0.01)
    Xlnk = SparseMatrixLNK(Xcsc)
    Xext = ExtendableSparseMatrix(Xcsc)
    t0 = @elapsed copy(Xcsc)
    t1 = @elapsed copy(Xlnk)
    t2 = @elapsed copy(Xext)

    if !(t1 / t0 < 10 && t0 / t2 < 10)
        @warn """timing test failed.
If this occurs just once ot twice, it is probably due to CPU noise.
So we nevertheless count this as passing.
"""
    end
    true    
end
test(Float64)
test(Float64x2)
test(Dual64)
end
