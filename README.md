# ExtendableSparse.jl

[![Build Status](https://img.shields.io/travis/j-fu/ExtendableSparse.jl/master.svg?label=Linux+MacOSX+Windows)](https://travis-ci.com/j-fu/ExtendableSparse.jl)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://j-fu.github.io/ExtendableSparse.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://j-fu.github.io/ExtendableSparse.jl/dev)



Sparse matrix class with efficient successive insertion of entries and entry update.

## Rationale
Without an intermediate data structure, efficient successive insertion/update of possibly duplicate entries in random order into a standard compressed column storage structure appears to be not possible. The package introduces `ExtendableSparseMatrix`, a delegating wrapper containing a Julia standard `SparseMatrixCSC` struct for performing linear algebra operations and a `SparseMatrixLNK` struct realising a linked list based (but realised in vectors) format collecting new entries.

The later is modeled after the linked list sparse matrix format described in the [whitepaper](https://www-users.cs.umn.edu/~saad/software/SPARSKIT/paper.ps) by Y. Saad. See also exercise P.3-16  in his [book](https://www-users.cs.umn.edu/~saad/IterMethBook_2ndEd.pdf).

Any linear algebra method on `ExtendableSparseMatrix` starts with a `flush!` method which adds the LNK entries and the existing CSC entries into a new CSC struct and resets the LNK struct.

`ExtendableSparseMatrix` is aimed to work as a drop-in replacement to `SparseMatrixCSC` in finite element and finite volume codes especally in those cases where the sparsity structure is hard to detect a priori and where working with an intermediadte COO representation appears to be not convenient.

## Caveat

This package assumes that a $m \times n$  matrix is sparse if *each* row and *each* column have less than $C$ entries with
$C << n$ and $C <<m$ . Adding a full matrix row will be a performance hit.


## Working with ForwardDiff

In particular, it cooperates with [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) when it comes to the assembly of a sparse jacobian. For a function 'f!(y,x)' returning it's result in a vector `y`, one can use e.g.
````
x=...
y=zeros(n)
dresult=DiffResults.DiffResult(zeros(n),ExtendableSparseMatrix(n,n))
x=ForwardDiff.jacobian!(dresult,f!,y,x)
jac=DiffResults.jacobian(dresult)
h=jac\x
````

However, without a priori information on sparsity, ForwardDiff calls element insertion for the full range of n^2 indices, 
leading to a O(n^2) scaling behavior due to the nevertheless necessary search operations, see  this [discourse thread](https://discourse.julialang.org/t/non-sorted-sparsematrixcsc/37133).

## updateindex!
In addition, the package provides a method `updateindex!(A,op,v,i,j)` for both `SparseMatrixCSC` and for `ExtendableSparse` which allows to update a matrix element with one index search instead of two. It allows to replace e.g. `A[i,j]+=v` by `updateindex!(A,+,v,i,j)`. The former operation is lowered to 
````
%1 = Base.getindex(A, 1, 2)
%2 = %1 + 3
Base.setindex!(A, %2, 1, 2)
````
triggering two index searches, one for `getindex!` and another one for `setindex!`.

See [Julia issue #15630](https://github.com/JuliaLang/julia/issues/15630) for a discussion on this.





