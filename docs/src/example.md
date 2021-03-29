# Examples & Benchmarks


## Matrix creation example
An `ExtendableSparseMatrix` can serve as a drop-in replacement for
`SparseMatrixCSC`, albeit with faster assembly during the buildup
phase when using index based access.

Let us define a simple assembly loop
```@example 1
using ExtendableSparse # hide
using SparseArrays     # hide
using BenchmarkTools   # hide
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1 # hide
function assemble(A)
    n=size(A,1)
    for i=1:n-1
        A[i+1,i]+=-1
        A[i,i+1]+=-1
        A[i,i]+=1
        A[i+1,i+1]+=1
    end
end;
```

Measure the time (in seconds) for assembling a SparseMatrixCSC:
```@example 1
t_csc= @belapsed begin
                   A=spzeros(10_000,10_000)
                   assemble(A)
                 end
```

An `ExtendableSparseMatrix` can be used as a drop-in replacement.
However, before any other use, this needs an internal
structure rebuild which is invoked by the flush! method.
```@example 1
t_ext=@belapsed  begin 
                   A=ExtendableSparseMatrix(10_000,10_000)
                   assemble(A)
                   flush!(A)
                 end
```
All  specialized methods of linear algebra functions (e.g. `\`)
for `ExtendableSparseMatrix`  call `flush!` before proceeding.

The overall time gain from using `ExtendableSparse` is:
```@example 1
t_ext/t_csc
```


The reason for this situation is that the `SparseMatrixCSC` struct
just contains the data for storing the matrix in the compressed
column format. Inserting a new entry in this storage scheme is
connected with serious bookkeeping and shifts of large portions
of array content.  

Julia provides the
[`sparse`](https://docs.julialang.org/en/v1/stdlib/SparseArrays/#SparseArrays.sparse)
method which  uses an intermediate  storage of  the data in  two index
arrays and a value array, the so called coordinate (or COO) format:

```@example 1
function assemble_coo(n)
    I=zeros(Int64,0)
    J=zeros(Int64,0)
    V=zeros(0)
    function update(i,j,v)
        push!(I,i)
        push!(J,j)
        push!(V,v)
    end
    for i=1:n-1
        update(i+1,i,-1)
        update(i,i+1,-1)
        update(i,i,1)
        update(i+1,i+1,1)
    end
    sparse(I,J,V)
end;

t_coo=@belapsed assemble_coo(10_000)
```

While more convenient to use, the assembly based on `ExtendableSparseMatrix` is only slightly
slower:

```@example 1
t_ext/t_coo
```



Below one finds a more elaborate discussion for a quasi-3D problem.

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

```@example 1
@belapsed fdrand(30,30,30, matrixtype=ExtendableSparseMatrix)
```

### Benchmark for  SparseMatrixCSC
Here, for comparison we use  a `SparseMatrixCSC` created with `spzeros` and insert
entries via `A[i,j]+=v`.

```@example
using ExtendableSparse # hide
using SparseArrays     # hide
using BenchmarkTools   # hide
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1 # hide
@belapsed fdrand(30,30,30, matrixtype=SparseMatrixCSC)
```

### Benchmark for  intermediate coordinate format
A `SparseMatrixCSC` is created by accumulating data into arrays `I`,`J`,`A` and
calling `sparse(I,J,A)`

```@example
using ExtendableSparse # hide
using SparseArrays     # hide
using BenchmarkTools   # hide
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1 # hide
@belapsed fdrand(30,30,30, matrixtype=:COO)
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
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1 # hide
@belapsed fdrand(30,30,30, 
    matrixtype=ExtendableSparseMatrix,
    update=(A,v,i,j)-> updateindex!(A,+,v,i,j))
```



## Matrix update benchmark
For repeated calculations on the same sparsity structure (e.g. for time dependent
problems or Newton iterations) it is convenient to skip all but the first creation steps
and to just replace the values in the matrix after setting the elements of the `nzval` 
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
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1 # hide
A=fdrand(30,30,30, matrixtype=SparseMatrixCSC)
@belapsed fdrand!(A,30,30,30, 
                   update=(A,v,i,j)-> A[i,j]+=v)
```

```@example
using ExtendableSparse # hide
using SparseArrays     # hide
using BenchmarkTools   # hide
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1 # hide
A=fdrand(30,30,30, matrixtype=SparseMatrixCSC)
@belapsed fdrand!(A,30,30,30, 
                   update=(A,v,i,j)-> updateindex!(A,+,v,i,j))
```

### Benchmark for `ExtendableSparseMatrix`
```@example
using ExtendableSparse # hide
using BenchmarkTools   # hide
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1 # hide
A=fdrand(30,30,30, matrixtype=ExtendableSparseMatrix)
@belapsed fdrand!(A,30,30,30, 
                   update=(A,v,i,j)-> A[i,j]+=v)
```

```@example
using ExtendableSparse # hide
using BenchmarkTools   # hide
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1 # hide
A=fdrand(30,30,30, matrixtype=ExtendableSparseMatrix)
@belapsed fdrand!(A,30,30,30, 
                   update=(A,v,i,j)-> updateindex!(A,+,v,i,j))
```

Note that the update process for `ExtendableSparse` may be slightly slower
than for `SparseMatrixCSC` due to the overhead which comes from checking
the presence of new entries.


