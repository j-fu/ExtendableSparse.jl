module test_pardiso
using Test
using ExtendableSparse
using SparseArrays
using LinearAlgebra
using Pardiso

include("test_lu.jl")

@test test_lu1(Float64,10,10,10,lufac=PardisoLU())
@test test_lu1(Float64,25,40,1,lufac=PardisoLU())
@test test_lu1(Float64,1000,1,1,lufac=PardisoLU())
                       
@test test_lu2(Float64,10,10,10,lufac=PardisoLU())
@test test_lu2(Float64,25,40,1,lufac=PardisoLU())
@test test_lu2(Float64,1000,1,1,lufac=PardisoLU())
# @test test_lu2(10,10,10,lufac=PardisoLU(mtype=2))
# @test test_lu2(25,40,1,lufac=PardisoLU(mtype=2))
# @test test_lu2(1000,1,1,lufac=PardisoLU(mtype=2))
end
