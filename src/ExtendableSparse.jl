module ExtendableSparse
using DocStringExtensions
using SparseArrays
using LinearAlgebra

include("extension.jl")
include("extendable.jl")
include("sprand.jl")

export SparseMatrixExtension,ExtendableSparseMatrix,flush!,nnz, sprand!,sprand_sdd!

export colptrs
end # module
