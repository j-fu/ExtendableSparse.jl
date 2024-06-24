"""
    $(TYPEDEF)

Abstract type for sparse matrix extension.

Subtypes T_ext must implement:

Constructor T_ext(m,n)
SparseArrays.nnz(ext::T_ext)
Base.size(ext::T_ext)


Base.sum(extmatrices::Vector{T_ext}, csx)
  - Add csx matrix and extension matrices (one per partition) and return csx matrix

rawupdateindex!(ext::Text, op, v, i, j) where {Tv, Ti}
  - Set ext[i,j]+=v, possibly insert entry into matrix.


Optional:

Base.+(ext::T_ext, csx)
  - Add extension matrix and csc/csr matrix, return csx matrix

"""
abstract type AbstractSparseMatrixExtension{Tv, Ti} <: AbstractSparseMatrix{Tv,Ti} end

Base.:+(ext::AbstractSparseMatrixExtension, csx) = sum([ext],csx) 
