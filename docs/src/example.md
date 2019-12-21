# Examples

An `ExtendableSparseMatrix` can serve as a drop-in replacement for
`SparseMatrixCSC`, albeit with faster assembly.

The code below provides a small benchmark example. 
```julia
using ExtendableSparse
using SparseArrays

function ext_create(n, row_nz)
    A=ExtendableSparseMatrix(n,n)
    sprand_sdd!(A,row_nz);
end

function csc_create(n, row_nz)
    A=spzeros(n,n)
    sprand_sdd!(A,row_nz);
end

csc_create(10,2);
ext_create(10,2);
n=65536
row_nz=5
@time Acsc=csc_create(n,row_nz);
nnz(Acsc)
@time Aext=ext_create(n,row_nz);
nnz(Aext)
b=rand(n);
@time Acsc*(Acsc\b)≈b
@time Aext*(Aext\b)≈b
```

The function [`sprand_sdd!`](@ref) fills a 
sparse matrix with random entries such that it becomes strictly diagonally
dominant and thus invertible and has a fixed number of nonzeros in
its rows. Its  bandwidth is bounded by 2*sqrt(n), therefore it 
resembles a typical matrix of a 2D piecewise linear FEM discretization.
For filling a matrix a, the method  conveniently albeit naively
uses just `a[i,j]=value`. This approach is considerably faster with 
the [`ExtendableSparseMatrix`](@ref).


