# Examples & Benchmarks


## Matrix creation example
An `ExtendableSparseMatrix` can serve as a drop-in replacement for
`SparseMatrixCSC`, albeit with faster assembly during the buildup
phase when using index based access. That means that code similar
to the following example should be fast enough to avoid the assembly
steps using the coordinate format:
    
```@example
using ExtendableSparse # hide
n=3
A=ExtendableSparseMatrix(n,n)
for i=1:n-1
    A[i,i+1]=i
end
A
```


## Matrix creation benchmark

The method [`fdrand`](@ref)  creates a matrix similar to the discretization
matrix of a Poisson equation on a d-dimensional cube. The approach is similar
to that of a typical finite element code: calculate a local stiffness matrix
and assemble it into the global one.


### Benchmark for [`ExtendableSparseMatrix`](@ref) 


The code uses the index access API for the creation of the matrix,
inserting elements via `A[i,j]+=v`,
using an intermediate linked list structure which upon return
ist flushed into a SparseMatrixCSC structure.

```@example
using ExtendableSparse # hide
using SparseArrays     # hide
using BenchmarkTools   # hide

t=@belapsed fdrand(30,30,30, matrixtype=ExtendableSparseMatrix)
```

### Benchmark for  SparseMatrixCSC
Here, for comparison we use  a `SparseMatrixCSC` created with `spzeros` and insert
entries via `A[i,j]+=v`.

```@example
using ExtendableSparse # hide
using SparseArrays     # hide
using BenchmarkTools   # hide

t=@belapsed fdrand(30,30,30, matrixtype=SparseMatrixCSC)
```

### Benchmark for  intermediate coordinate format
A `SparseMatrixCSC` is created by accumulating data into arrays `I`,`J`,`A` and
calling `sparse(I,J,A)`

```@example
using ExtendableSparse # hide
using SparseArrays     # hide
using BenchmarkTools   # hide

t=@belapsed fdrand(30,30,30, matrixtype=:COO)
```

This is nearly on par with matrix creation via `ExtendableSparseMatrix`, but the
later can be made faster:


### Benchmark  for `ExtendableSparseMatrix` with `updateindex`
Here, we use  a `ExtendableSparseMatrix created with `spzeros` and insert
entries via `updateindex(A,+,v,i,j)`, see the discussion below.

```@example
using ExtendableSparse # hide
using SparseArrays     # hide
using BenchmarkTools   # hide

t=@belapsed fdrand(30,30,30, 
    matrixtype=ExtendableSparseMatrix,
    update=(A,v,i,j)-> updateindex!(A,+,v,i,j))
```



## Matrix update benchmark
For repeated calculations on the same sparsity structure (e.g. for time dependent
problems or Newton iterations) it is convenient to skip all but the first creation steps
and to just replace the values in the matrix after setting then elements of the `nzval` 
vector to zero. Typically in finite element and finite volume methods this step updates
matrix entries (most of them several times) by adding values. In this case, the current indexing
interface of Julia requires to access the matrix twice:

```@example
using SparseArrays     # hide
A=spzeros(3,3)
Meta.@lower A[1,2]+=3
```
For sparse matrices this requires to perform the index search in the structure twice.
The packages provides the method [`updateindex!`](@ref) for both `SparseMatrixCSC` and 
for `ExtendableSparse` which allows to update a matrix element with just one index search.


### Benchmark for `SparseMatrixCSC`
```@example
using ExtendableSparse # hide
using SparseArrays     # hide
using BenchmarkTools   # hide

A=fdrand(30,30,30, matrixtype=SparseMatrixCSC)
t=@belapsed fdrand!(A,30,30,30, 
                   update=(A,v,i,j)-> A[i,j]+=v)
```

```@example
using ExtendableSparse # hide
using SparseArrays     # hide
using BenchmarkTools   # hide

A=fdrand(30,30,30, matrixtype=SparseMatrixCSC)
t=@belapsed fdrand!(A,30,30,30, 
                   update=(A,v,i,j)-> updateindex!(A,+,v,i,j))
```

### Benchmark for `ExtendableSparseMatrix`
```@example
using ExtendableSparse # hide
using BenchmarkTools   # hide

A=fdrand(30,30,30, matrixtype=ExtendableSparseMatrix)
t=@belapsed fdrand!(A,30,30,30, 
                   update=(A,v,i,j)-> A[i,j]+=v)
```

```@example
using ExtendableSparse # hide
using BenchmarkTools   # hide

A=fdrand(30,30,30, matrixtype=ExtendableSparseMatrix)
t=@belapsed fdrand!(A,30,30,30, 
                   update=(A,v,i,j)-> updateindex!(A,+,v,i,j))
```

Note that the update process for `ExtendableSparse` may be slightly slower
than for `SparseMatrixCSC` due to the overhead which comes from checking
the presence of new entries.


