module ExtendableSparse
using DocStringExtensions
using SparseArrays
using LinearAlgebra

include("extension.jl")
include("extendable.jl")

export SparseMatrixExtension,ExtendableSparseMatrix,flush!,nnz

export xcolptrs,colptrs
end # module
