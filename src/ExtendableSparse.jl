module ExtendableSparse
using DocStringExtensions
using SparseArrays
using LinearAlgebra

include("sparsematrixlnk.jl")
include("extendable.jl")
include("preconditioners.jl")
include("sprand.jl")

export SparseMatrixLNK,ExtendableSparseMatrix,flush!,nnz, sprand!,sprand_sdd!
export JacobiPreconditioner, ILU0Preconditioner, update!

export colptrs
end # module
