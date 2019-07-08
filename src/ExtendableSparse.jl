module ExtendableSparse
using DocStringExtensions
using SparseArrays
using LinearAlgebra

include("extension.jl")
include("extendable.jl")
include("sprand.jl")

export SparseMatrixExtension,ExtendableSparseMatrix,flush!,nnz,sprand!

export xcolptrs,colptrs
end # module
