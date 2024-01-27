module test_constructors
using Test
using LinearAlgebra
using ExtendableSparse
using SparseArrays
using Random
using MultiFloats
using ForwardDiff

const Dual64 = ForwardDiff.Dual{Float64, Float64, 1}

function test_construct(T)
    m = ExtendableSparseMatrix(T, 10, 10)
    eltype(m) == T
end

function test_sprand(T)
    m = ExtendableSparseMatrix(sprand(T, 10, 10, 0.1))
    eltype(m) == T
end

function test_transient_construction(; m = 10, n = 10, d = 0.1)
    csc = sprand(m, n, d)
    lnk = SparseMatrixLNK(csc)
    csc2 = SparseMatrixCSC(lnk)
    csc2 == csc
end

function test()
    m1 = ExtendableSparseMatrix(10, 10)
    @test eltype(m1) == Float64
    @test test_construct(Float16)
    @test test_construct(Float32)
    @test test_construct(Float64x2)
    @test test_construct(Dual64)

    acsc = sprand(10, 10, 0.5)
    @test sparse(ExtendableSparseMatrix(acsc)) == acsc

    D = Diagonal(rand(10))
    ED = ExtendableSparseMatrix(D)
    @test all([D[i, i] == ED[i, i] for i = 1:10])

    I = rand(1:10, 100)
    J = rand(1:10, 100)
    A = rand(100)
    @test sparse(I, J, A) == sparse(ExtendableSparseMatrix(I, J, A))

    @test test_sprand(Float16)
    @test test_sprand(Float32)
    @test test_sprand(Float64)
    @test test_sprand(Float64x2)
    @test test_sprand(Dual64)

    for irun = 1:10
        m = rand((1:1000))
        n = rand((1:1000))
        d = 0.3 * rand()
        @test test_transient_construction(m = m, n = n, d = d)
    end
end
test()
end
