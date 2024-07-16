using Test
using LinearAlgebra
using SparseArrays
using ExtendableSparse
using ExtendableSparse.Experimental
using Printf
using BenchmarkTools

using MultiFloats
using ForwardDiff
using ExplicitImports

@testset "ExplicitImports" begin
    @test ExplicitImports.check_no_implicit_imports(ExtendableSparse, allow_unanalyzable=(ExtendableSparse.Experimental,)) === nothing
    @test ExplicitImports.check_no_stale_explicit_imports(ExtendableSparse, allow_unanalyzable=(ExtendableSparse.Experimental,)) === nothing
end

@testset "Parallel" begin
    include("test_parallel.jl")

    for Tm in [STExtendableSparseMatrixCSC, MTExtendableSparseMatrixCSC, ExtendableSparseMatrix]
        for N in [10000,20000]
            test_parallel.test_correctness_build_seq(N,Tm, dim=2)
        end
    end

    for Tm in [MTExtendableSparseMatrixCSC]
        for N in [10000,20000]
            test_parallel.test_correctness_update(N,Tm, dim=2)
            test_parallel.test_correctness_build(N,Tm, dim=2)
            test_parallel.test_correctness_mul(N,Tm,dim=2)
        end
    end
end

@testset "ExperimentalParallel" begin
    include("ExperimentalParallel.jl")
    for d=[1,2,3]
        for N in [10,rand(30:40),50]
            ExperimentalParallel.test_correctness_build(N,d)
        end
    end
end


@testset "Constructors" begin include("test_constructors.jl") end

@testset "Copy-Methods" begin include("test_copymethods.jl") end

@testset "Updates" begin include("test_updates.jl") end

@testset "Assembly" begin include("test_assembly.jl") end

@testset "Construction timings" begin include("test_timings.jl") end

@testset "Operations" begin include("test_operations.jl") end

@testset "fdrand" begin include("test_fdrand.jl") end

@testset "Backslash" begin include("test_backslash.jl") end

@testset "Dirichlet" begin include("test_dirichlet.jl") end

@testset "LinearSolve" begin include("test_linearsolve.jl") end

@testset "Preconditioners" begin include("test_preconditioners.jl") end

@testset "Symmetric" begin include("test_symmetric.jl") end

@testset "ExtendableSparse.LUFactorization" begin include("test_default_lu.jl") end

@testset "Sparspak" begin include("test_sparspak.jl") end

if ExtendableSparse.USE_GPL_LIBS
    @testset "Cholesky" begin include("test_default_cholesky.jl") end
end

@testset "block" begin include("test_block.jl") end

#@testset "parilu0" begin include("test_parilu0.jl") end


# @testset "mkl-pardiso" begin if !Sys.isapple()
#     include("test_mklpardiso.jl")
# end end


# if Pardiso.PARDISO_LOADED[]
#      @testset "pardiso" begin include("test_pardiso.jl") end
# end
