module ExtendableSparse
using DocStringExtensions
using SparseArrays
using LinearAlgebra

include("sparsematrixcsc.jl")
include("sparsematrixlnk.jl")
include("extendable.jl")
include("preconditioners.jl")
include("sprand.jl")

export SparseMatrixLNK,ExtendableSparseMatrix,flush!,nnz, sprand!,sprand_sdd!, fdrand,fdrand!
export JacobiPreconditioner, ILU0Preconditioner, updateindex!

export colptrs
end # module
