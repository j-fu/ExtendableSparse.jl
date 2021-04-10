module ExtendableSparse
using SparseArrays
using LinearAlgebra
using SuiteSparse
using Requires

using DocStringExtensions



include("sparsematrixcsc.jl")
include("sparsematrixlnk.jl")
include("extendable.jl")

export SparseMatrixLNK,ExtendableSparseMatrix,flush!,nnz, updateindex!, rawupdateindex!, colptrs

include("factorizations.jl")
export JacobiPreconditioner, ILU0Preconditioner, ParallelJacobiPreconditioner
export issolver
export factorize,factorize!, update!

include("simple_iteration.jl")
export simple,simple!




include("sprand.jl")
export sprand!,sprand_sdd!, fdrand,fdrand!,fdrand_coo


function __init__()
    @require Pardiso = "46dd5b70-b6fb-5a00-ae2d-e8fea33afaf2" include("pardiso_lu.jl")
    @require IncompleteLU = "40713840-3770-5561-ab4c-a76e7d0d7895" include("ilut.jl")
    @require AlgebraicMultigrid = "2169fc97-5a83-5252-b627-83903c6c433c" include("amg.jl")
end



end # module
