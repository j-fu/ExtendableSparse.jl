"""
    $(TYPEDEF)

Abstract type for sparse matrix extension.

Subtypes T_ext must implement:

- Constructor `T_ext(m,n)`
- `SparseArrays.nnz(ext::T_ext)`
- `Base.size(ext::T_ext)`
- `Base.sum(extmatrices::Vector{T_ext}, csx)`:  add csr or csc matrix and extension matrices (one per partition) and return csx matrix
- `Base.+(ext::T_ext, csx)` (optional)  - Add extension matrix and csc/csr matrix, return csx matrix
- `rawupdateindex!(ext::Text, op, v, i, j, tid) where {Tv, Ti}`: Set `ext[i,j]op=v`, possibly insert new entry into matrix. `tid` is a
task or partition id

"""
abstract type AbstractSparseMatrixExtension{Tv, Ti} <: AbstractSparseMatrix{Tv,Ti} end

Base.:+(ext::AbstractSparseMatrixExtension, csx) = sum([ext],csx) 
