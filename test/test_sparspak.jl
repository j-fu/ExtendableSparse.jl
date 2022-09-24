module test_sparspak

using Test
using ExtendableSparse
using SparseArrays
using LinearAlgebra


include("test_lu.jl")

for T in [Float64, Float64x1, Float64x2, Dual64]
    println("$T:")
    @test test_lu1(T,10,10,10,lufac=SparspakLU(valuetype=T))
    @test test_lu1(T,25,40,1,lufac=SparspakLU(valuetype=T))
    @test test_lu1(T,1000,1,1,lufac=SparspakLU(valuetype=T))
    
    @test test_lu2(T,10,10,10,lufac=SparspakLU(valuetype=T))
    @test test_lu2(T,25,40,1,lufac=SparspakLU(valuetype=T))
    @test test_lu2(T,1000,1,1,lufac=SparspakLU(valuetype=T))
end
end
