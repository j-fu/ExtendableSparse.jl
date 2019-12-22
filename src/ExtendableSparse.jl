module ExtendableSparse
using DocStringExtensions
using SparseArrays
using LinearAlgebra

include("sparsematrixlnk.jl")
include("extendable.jl")
include("sprand.jl")

export SparseMatrixLNK,ExtendableSparseMatrix,flush!,nnz, sprand!,sprand_sdd!

export colptrs
end # module
