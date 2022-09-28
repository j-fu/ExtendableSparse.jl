module test_constructors
using Test
using ExtendableSparse
using SparseArrays
using Random
using MultiFloats
using ForwardDiff

const Dual64=ForwardDiff.Dual{Float64,Float64,1}
Random.rand(rng::AbstractRNG, ::Random.SamplerType{ForwardDiff.Dual{T,V,N}}) where {T,V,N} = ForwardDiff.Dual{T,V,N}(rand(rng,T))

function test_construct(T)
    m=ExtendableSparseMatrix(T,10,10)
    @test eltype(m)==T
end


function test_sprand(T)
    m=ExtendableSparseMatrix(sprand(T,10,10,0.1))
    @test eltype(m)==T
end

function test_transient_construction(;m=10,n=10,d=0.1) where {Tv,Ti}
    csc=sprand(m,n,d)
    lnk=SparseMatrixLNK(csc)
    csc2=SparseMatrixCSC(lnk)
    csc2==csc
end


function test()
    m1=ExtendableSparseMatrix(10,10)
    @test eltype(m1)==Float64
    test_construct(Float16)
    test_construct(Float32)
    test_construct(Float64x2)
    test_construct(Dual64)

    test_sprand(Float16)
    test_sprand(Float32)
    test_sprand(Float64)
    test_sprand(Float64x2)
    test_sprand(Dual64)

    for irun=1:10
        m=rand((1:1000))
        n=rand((1:1000))
        d=0.3*rand()
        @test test_transient_construction(m=m,n=n,d=d)
    end


end
test()
end
