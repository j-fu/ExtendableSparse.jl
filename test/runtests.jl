using Test
using LinearAlgebra
using SparseArrays
using ExtendableSparse
using Printf
using BenchmarkTools

using Pardiso
using Sparspak
using MultiFloats
using ForwardDiff


@testset "Constructors" begin
    include("test_constructors.jl")
end

@testset "Copy-Methods" begin
    include("test_copymethods.jl")
end

@testset "Updates" begin
    include("test_updates.jl")
end

@testset "Assembly" begin
    include("test_assembly.jl")
end

@testset "Construction timings" begin
    include("test_timings.jl")
end

@testset "Operations" begin
    include("test_operations.jl")
end

@testset "fdrand" begin
    include("test_fdrand.jl")
end

if Base.USE_GPL_LIBS
    #test requires SuiteSparse which is not available on non-GPL builds
    @testset "Preconditioners" begin
        include("test_preconditioners.jl")
    end
end

@testset "Symmetric" begin
    include("test_symmetric.jl")
end


if Base.USE_GPL_LIBS
#requires SuiteSparse which is not available on non-GPL builds
@testset "ExtendableSparse.LUFactorization" begin
    include("test_default_lu.jl")
end
end

if Base.USE_GPL_LIBS
#requires SuiteSparse which is not available on non-GPL builds
@testset "Cholesky" begin
    include("test_default_cholesky.jl")
end
end


@testset "mkl-pardiso" begin
    if !Sys.isapple()
        include("test_mklpardiso.jl")
    end
end

if Pardiso.PARDISO_LOADED[]
    @testset "pardiso" begin
        include("test_pardiso.jl")
    end
end


@testset "sparspak" begin
    include("test_sparspak.jl")
end

@testset "parilu0" begin
    include("test_parilu0.jl")
end



