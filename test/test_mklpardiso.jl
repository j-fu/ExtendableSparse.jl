module test_mklpardiso
using Test
using ExtendableSparse
using SparseArrays
using LinearAlgebra
using Pardiso

include("test_lu.jl")
Base.eps(ComplexF64)=eps(Float64)

@test test_lu1(Float64, 10, 10, 10, lufac = MKLPardisoLU())
@test test_lu1(Float64, 25, 40, 1, lufac = MKLPardisoLU())
@test test_lu1(Float64, 1000, 1, 1, lufac = MKLPardisoLU())

@test test_lu1(ComplexF64, 10, 10, 10, lufac = MKLPardisoLU())
@test test_lu1(ComplexF64, 25, 40, 1, lufac = MKLPardisoLU())
@test test_lu1(ComplexF64, 1000, 1, 1, lufac = MKLPardisoLU())



@test test_lu2(Float64, 10, 10, 10, lufac = MKLPardisoLU())
@test test_lu2(Float64, 25, 40, 1, lufac = MKLPardisoLU())
@test test_lu2(Float64, 1000, 1, 1, lufac = MKLPardisoLU())


end
