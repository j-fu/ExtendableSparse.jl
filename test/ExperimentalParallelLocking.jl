module ExperimentalParallelLocking

using ExtendableSparse,SparseArrays
using ExtendableSparse: with_locking!
using ExtendableSparse.Experimental
using BenchmarkTools
using Test

"""
    test_correctness_update(N)

Test correctness of parallel assembly on NxN grid  during 
update phase, assuming that the structure has been assembled.
"""
function test_correctness_update(N)
    with_locking!(true)
    X=1:N
    Y=1:N
    A=ExtendableSparseMatrix(N^2,N^2)
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
    with_locking!(false)
end

"""
    test_correctness_build(N)

Test correctness of parallel assembly on NxN grid  during 
build phase, assuming that no structure has been assembled.
"""
function test_correctness_build(N)
    with_locking!(true)
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
        A=ExtendableSparseMatrix(N^2,N^2)
        partassemble!(A,X,Y, np)
        @test nonzeros(A)≈nz
    end
    with_locking!(false)
end



"""
    speedup_update(N)

Benchmark parallel speedup of update phase of parallel assembly on NxN grid.
Check for correctness as well.
"""
function speedup_update(N; allnp=[4,5,6,7,8,9,10])
    with_locking!(true)
    X=1:N
    Y=1:N
    A=ExtendableSparseMatrix(N^2,N^2)
    partassemble!(A,X,Y)
    nz=copy(nonzeros(A))
    # Get the base timing
    # During setup, set matrix entries to zero while keeping  the structure 
    t0=@belapsed partassemble!($A,$X,$Y) seconds=1 setup=(nonzeros($A).=0)
    result=[]
    for np in allnp
        # Get the parallel timing
        # During setup, set matrix entries to zero while keeping  the structure 
        t=@belapsed partassemble!($A,$X,$Y,$np) seconds=1 setup=(nonzeros($A).=0)
        @assert nonzeros(A)≈nz
        push!(result,(np,round(t0/t,digits=2)))
    end
    with_locking!(false)
    result
end


"""
    speedup_build(N)

Benchmark parallel speedup of structure build phase of parallel assembly on NxN grid.
Check for correctness as well.

Works in the moment with locking.
"""
function speedup_build(N; allnp=[4,5,6,7,8,9,10])
    with_locking!(true)
    X=1:N
    Y=1:N
    A=ExtendableSparseMatrix(N^2,N^2)
    partassemble!(A,X,Y)
    nz=copy(nonzeros(A))
    reset!(A)
    partassemble!(A,X,Y)
    @assert nonzeros(A)≈(nz)
    
    # Get the base timing
    # During setup, reset matrix to empty state.
    t0=@belapsed partassemble!($A,$X,$Y) seconds=1 setup=(reset!($A))
    
    result=[]
    for np in allnp
        # Get the parallel timing
        # During setup, reset matrix to empty state.
        t=@belapsed partassemble!($A,$X,$Y,$np) seconds=1 setup=(reset!($A))
        @assert nonzeros(A)≈nz
        push!(result,(np,round(t0/t,digits=2)))
    end
    with_locking!(false)
    result
end
end
