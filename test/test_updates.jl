module test_updates
using Test
using ExtendableSparse
using SparseArrays
using Random
using MultiFloats
using ForwardDiff
const Dual64 = ForwardDiff.Dual{Float64, Float64, 1}

function test(T)
    A = ExtendableSparseMatrix(T, 10, 10)
    @test nnz(A) == 0
    A[1, 3] = 5
    updateindex!(A, +, 6.0, 4, 5)
    updateindex!(A, +, 0.0, 2, 3)
    @test nnz(A) == 2
    rawupdateindex!(A, +, 0.0, 2, 3)
    @test nnz(A) == 3
    dropzeros!(A)
    @test nnz(A) == 2
    rawupdateindex!(A, +, 0.1, 2, 3)
    @test nnz(A) == 3
    dropzeros!(A)
    @test nnz(A) == 3
end

test(Float64)
test(Float64x2)
test(Dual64)
end
