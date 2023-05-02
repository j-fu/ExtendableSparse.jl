module test_block
using Test
using ExtendableSparse
using ILUZero, AlgebraicMultigrid
using IterativeSolvers
using LinearAlgebra
using Sparspak


ExtendableSparse.needs_copywrap(::typeof(ilu0))=false

function main(;n=100)
    
    A=fdrand(n,n)
    parts=[1:2:n^2, 2:2:n^2]
    sol0=ones(n^2)
    b=A*ones(n^2);
    sol=cg(A,b,Pl=ilu0(A))
    
    @test sol≈sol0
    
    sol=cg(A,b,Pl=BlockPreconditioner(A;parts))
    @test sol≈sol0

    
    sol=cg(A,b,Pl=BlockPreconditioner(A;parts, factorization=ilu0))
    @test sol≈sol0
    
    sol=cg(A,b,Pl=BlockPreconditioner(A;parts, factorization=ILUZeroPreconditioner))
    @test sol≈sol0
    
    sol=cg(A,b,Pl=BlockPreconditioner(A;parts, factorization=ILU0Preconditioner))
    @test sol≈sol0

    sol=cg(A,b,Pl=BlockPreconditioner(A;parts, factorization=JacobiPreconditioner))
    @test sol≈sol0


    sol=cg(A,b,Pl=BlockPreconditioner(A;parts, factorization=AMGPreconditioner))
    @test sol≈sol0

    sol=cg(A,b,Pl=BlockPreconditioner(A;parts, factorization=sparspaklu))
    @test sol≈sol0

    sol=cg(A,b,Pl=BlockPreconditioner(A;parts, factorization=SparspakLU))
    @test sol≈sol0

end

main(n=100)
end
