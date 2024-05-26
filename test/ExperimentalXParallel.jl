module ExperimentalXParallel

using ExtendableSparse,SparseArrays, ExtendableSparse.Experimental
using BenchmarkTools
using Test


function test_correctness_update(N,Tm::Type{<:AbstractSparseMatrix})
    X=1:N
    Y=1:N
    A=Tm{Float64,Int}(N^2,N^2,1)
    allnp=[4,5,6,7,8]

    # Assembele without partitioning
    # this gives the "base truth" to compare with
    partassemble!(A,X,Y)

    # Save the nonzeros 
    nz=copy(nonzeros(A))
    for np in allnp
        # Reset the nonzeros, keeping the structure intact
        nonzeros(A).=0
        # Parallel assembly whith np threads
        partassemble!(A,X,Y, np)
        @test nonzeros(A)≈nz
    end
end

"""
    test_correctness_build(N)

Test correctness of parallel assembly on NxN grid  during 
build phase, assuming that no structure has been assembled.
"""
function test_correctness_build(N,Tm::Type{<:AbstractSparseMatrix})
    X=1:N
    Y=1:N
    allnp=[4,5,6,7,8]
    # Get the "ground truth"
    A=ExtendableSparseMatrix(N^2,N^2)
    partassemble!(A,X,Y)
    nz=copy(nonzeros(A))
    for np in allnp
        # Make a new matrix and assemble parallel.
        # this should result in the same nonzeros
        A=Tm(N^2,N^2,1)
        partassemble!(A,X,Y, np)
        @test nonzeros(A)≈nz
    end
end

function test_correctness_mul(N,Tm::Type{<:AbstractSparseMatrix};     allnp=[4,5,6,7,8])
    X=1:N
    Y=1:N
    A0=ExtendableSparseMatrix(N^2,N^2)
    partassemble!(A0,X,Y)

    for np in allnp
        A=Tm(N^2,N^2,1)
        partassemble!(A,X,Y,np)
        b=rand(N^2)
        @test A*b ≈ A0*b
    end    
end

function speedup_update(N,Tm::Type{<:AbstractSparseMatrix}; allnp=[4,5,6,7,8,9,10])
    X=1:N
    Y=1:N
    A=ExtendableSparseMatrix(N^2,N^2)
    partassemble!(A,X,Y)
    nz=copy(nonzeros(A))
    # Get the base timing
    # During setup, set matrix entries to zero while keeping  the structure 
    t0=@belapsed partassemble!($A,$X,$Y) seconds=1 setup=(nonzeros($A).=0)
    result=[]
    A=Tm(N^2,N^2,1)
    for np in allnp
        # Get the parallel timing
        # During setup, set matrix entries to zero while keeping  the structure
        partassemble!(A,X,Y,np)
        t=@belapsed partassemble!($A,$X,$Y,$np,reset=false) seconds=1 setup=(nonzeros($A).=0)
        @assert nonzeros(A)≈nz
        push!(result,(np,round(t0/t,digits=2)))
    end
    result
end

function speedup_build(N,Tm::Type{<:AbstractSparseMatrix}; allnp=[4,5,6,7,8,9,10])
    X=1:N
    Y=1:N
    A0=ExtendableSparseMatrix(N^2,N^2)
    A=Tm(N^2,N^2,1)
    partassemble!(A0,X,Y)
    nz=copy(nonzeros(A0))
    reset!(A0)
    partassemble!(A0,X,Y)
    @assert nonzeros(A0)≈(nz)

    partassemble!(A,X,Y)
    nz=copy(nonzeros(A))
    reset!(A)
    partassemble!(A,X,Y)
    @assert nonzeros(A)≈(nz)

    # Get the base timing
    # During setup, reset matrix to empty state.
    t0=@belapsed partassemble!($A0,$X,$Y) seconds=1 setup=(reset!($A0))
    
    result=[]
    for np in allnp
        # Get the parallel timing
        # During setup, reset matrix to empty state.
        t=@belapsed partassemble!($A,$X,$Y,$np) seconds=1 setup=(reset!($A))
        @assert nonzeros(A)≈nz
        push!(result,(np,round(t0/t,digits=2)))
    end
    result
end

function speedup_mul(N,Tm::Type{<:AbstractSparseMatrix}; allnp=[4,5,6,7,8,9,10])
    X=1:N
    Y=1:N
    
    A0=ExtendableSparseMatrix(N^2,N^2)
    partassemble!(A0,X,Y)
    b=rand(N^2)
    t0=@belapsed $A0*$b seconds=1
    
    result=[]
    for np in allnp
        A=Tm(N^2,N^2,1)
        partassemble!(A,X,Y,np)
        t=@belapsed $A*$b seconds=1
        push!(result,(np,round(t0/t,digits=2)))
    end
    result
end

end

