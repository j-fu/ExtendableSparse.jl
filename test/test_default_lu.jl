module test_default_lu
using Test
using ExtendableSparse
using SparseArrays
using LinearAlgebra

include("test_lu.jl")

@test test_lu1(Float64, 10, 10, 10)
@test test_lu1(Float64, 25, 40, 1)
@test test_lu1(Float64, 1000, 1, 1)

@test test_lu2(Float64, 10, 10, 10)
@test test_lu2(Float64, 25, 40, 1)
@test test_lu2(Float64, 1000, 1, 1)
end
