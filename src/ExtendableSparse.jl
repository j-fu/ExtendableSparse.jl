module ExtendableSparse
using DocStringExtensions
using SparseArrays
using LinearAlgebra

include("sparsematrixcsc.jl")
include("sparsematrixlnk.jl")
include("extendable.jl")
export SparseMatrixLNK,ExtendableSparseMatrix,flush!,nnz, updateindex!, rawupdateindex!, colptrs

include("preconditioners.jl")
export JacobiPreconditioner, ILU0Preconditioner, ParallelJacobiPreconditioner
export ExtendablesSparseLU
export update!

include("simple_iteration.jl")
export simple,simple!

include("sprand.jl")
export sprand!,sprand_sdd!, fdrand,fdrand!,fdrand_coo
end # module
