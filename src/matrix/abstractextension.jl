"""
    $(TYPEDEF)

Abstract type for sparse matrix extension.

Subtypes T_ext must implement:


Constructor T_ext(m,n)
SparseArrays.nnz(ext::T_ext)
Base.size(ext::T_ext)

Base.+(ext::T_ext, csc)
  - Add extension matrix and csc matrix, return csc matrix

sum!(nodeparts::Vector{Ti},extmatrices::Vector{T_ext}, cscmatrix)
  - Add csc matrix and extension matrices (one per partition) and return csc matrix
  - Fill nodeparts (already initialized at input) with information which partition was used to assemble node.
    i.e. if entry [i,j] comes from extmatrixes[p], set nodeparts[j]=p .

    This information may be used by matrix-vector multiplication and preconditioners

rawupdateindex!(ext::Text, op, v, i, j) where {Tv, Ti}
  - Set ext[i,j]+=v, possibly insert entry into matrix.


"""
abstract type AbstractSparseMatrixExtension{Tv, Ti} <: AbstractSparseMatrix{Tv,Ti} end
