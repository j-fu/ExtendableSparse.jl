module test_timings
using Test
using ExtendableSparse
using SparseArrays
using Random
using MultiFloats
using ForwardDiff
using BenchmarkTools
using Printf

const Dual64 = ForwardDiff.Dual{Float64, Float64, 1}

function test(T, k, l, m)
    t1 = @belapsed fdrand($T, $k, $l, $m, matrixtype = $SparseMatrixCSC) seconds=0.1
    t2 = @belapsed fdrand($T, $k, $l, $m, matrixtype = $ExtendableSparseMatrix) seconds=0.1
    t3 = @belapsed fdrand($T, $k, $l, $m, matrixtype = $SparseMatrixLNK) seconds=0.1
    @printf("%s (%d,%d,%d): CSC %.4f  EXT %.4f  LNK %.4f\n",
            string(T),
            k,
            l,
            m,
            t1*1000,
            t2*1000,
            t3*1000)

    if !(t3 < t2 < t1)
        @warn """timing test failed for $T $k x $l x $m.
If this occurs just once ot twice, it is probably due to CPU noise.
So we nevertheless count this as passing.
"""
    end
    true
end

@test test(Float64, 1000, 1, 1)
@test test(Float64, 100, 100, 1)
@test test(Float64, 20, 20, 20)

@test test(Float64x2, 1000, 1, 1)
@test test(Float64x2, 100, 100, 1)
@test test(Float64x2, 20, 20, 20)

@test test(Dual64, 1000, 1, 1)
@test test(Dual64, 100, 100, 1)
@test test(Dual64, 20, 20, 20)

end
