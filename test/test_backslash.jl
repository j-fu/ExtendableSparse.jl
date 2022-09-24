module test_backslash
using Test
using ExtendableSparse
using ForwardDiff
using MultiFloats
f64(x::ForwardDiff.Dual{T}) where T=Float64(ForwardDiff.value(x))
f64(x::Number)=Float64(x)
const Dual64=ForwardDiff.Dual{Float64,Float64,1}
Base.zero(TT::Type{ForwardDiff.Dual{T,T,N}}) where{T,N}=TT(zero(T))

function test_bslash(T,k,l,m)
    A=fdrand(T,k,l,m,rand=()->one(T),matrixtype=ExtendableSparseMatrix)
    b=ones(T,size(A,1))
    f64(sum(A\b))
end


function test_bslash(T)
    @info "$T:" 
    @test isapprox( test_bslash(T,100,1,1),5808.500000000794)
    @test isapprox( test_bslash(T,20,20,1), 45515.68342217646)
    @test isapprox( test_bslash(T,10,10,10), 185220.93903056852)
end

test_bslash(Float64)
test_bslash(Float64x1)
test_bslash(ForwardDiff.Dual{Float64,Float64,1})

end
