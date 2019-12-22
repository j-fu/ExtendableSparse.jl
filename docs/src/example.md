# Examples


## Usage idea
An `ExtendableSparseMatrix` can serve as a drop-in replacement for
`SparseMatrixCSC`, albeit with faster assembly.

    
```jldoctest
using ExtendableSparse
function example(n,bandwidth)
    A=ExtendableSparseMatrix(n,n);
    aii=0.0
    for i=1:n
        for j=max(1,i-bandwidth):5:min(n,i+bandwidth)
            aij=-rand()
            A[i,j]=aij
            aii+=abs(aij)
        end
        A[i,i]=aii+0.1
    end
    b=rand(n)
    x=A\b
    A*x≈ b
end
example(1000,100)
# output
true
```


## Benchmark
The code below provides a small benchmark example.
```julia
using ExtendableSparse
using SparseArrays

function ext_create(n)
    A=ExtendableSparseMatrix(n,n)
    sprand_sdd!(A);
end

function csc_create(n)
    A=spzeros(n,n)
    sprand_sdd!(A);
end

# Trigger JIT compilation before timing
csc_create(10);
ext_create(10);

n=90000
@time Acsc=csc_create(n);
nnz(Acsc)
@time Aext=ext_create(n);
nnz(Aext)
b=rand(n);
@time Acsc*(Acsc\b)≈b
@time Aext*(Aext\b)≈b
```

The function [`sprand_sdd!`](@ref) fills a 
sparse matrix with random entries such that it becomes strictly diagonally
dominant and thus invertible and has a fixed number of nonzeros in
its rows (similar to the method in the first example). Its  bandwidth is bounded by 2*sqrt(n), therefore it 
resembles a typical matrix of a 2D piecewise linear FEM discretization.
For filling a matrix a, the method  conveniently albeit naively
uses just `a[i,j]=value`. This approach is considerably faster with 
the [`ExtendableSparseMatrix`](@ref) which uses a linked list based
structure  [`SparseMatrixLNK`](@ref) to grab new entries.



