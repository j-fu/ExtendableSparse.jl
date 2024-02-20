module ExtendableSparse
using SparseArrays,StaticArrays
using LinearAlgebra
using Sparspak
using ILUZero

using Metis
using Base.Threads
using ExtendableGrids

if  !isdefined(Base, :get_extension)
    using Requires
end

# Define our own constant here in order to be able to
# test things at least a little bit..
const USE_GPL_LIBS = Base.USE_GPL_LIBS

if USE_GPL_LIBS
    using SuiteSparse
end

using DocStringExtensions

import SparseArrays: AbstractSparseMatrixCSC, rowvals, getcolptr, nonzeros

include("matrix/sparsematrixcsc.jl")
include("matrix/sparsematrixlnk.jl")
include("matrix/extendable.jl")

export SparseMatrixLNK,
       ExtendableSparseMatrix, flush!, nnz, updateindex!, rawupdateindex!, colptrs, sparse

export eliminate_dirichlet, eliminate_dirichlet!, mark_dirichlet


include("matrix/ExtendableSparseMatrixParallel/ExtendableSparseParallel.jl")

export ExtendableSparseMatrixParallel, SuperSparseMatrixLNK
export addtoentry!, reset!, dummy_assembly!, preparatory_multi_ps_less_reverse, fr, addtoentry!, rawupdateindex!, updateindex!, compare_matrices_light


include("factorizations/factorizations.jl")

export JacobiPreconditioner,
    ILU0Preconditioner,
    ILUZeroPreconditioner,
    PointBlockILUZeroPreconditioner,
    ParallelJacobiPreconditioner,
    ParallelILU0Preconditioner,
    BlockPreconditioner,allow_views,
    reorderlinsys

export AbstractFactorization, LUFactorization, CholeskyFactorization, SparspakLU
export issolver
export factorize!, update!

include("factorizations/simple_iteration.jl")
export simple, simple!

include("matrix/sprand.jl")
export sprand!, sprand_sdd!, fdrand, fdrand!, fdrand_coo, solverbenchmark


@static if  !isdefined(Base, :get_extension)
    function __init__()
        @require Pardiso = "46dd5b70-b6fb-5a00-ae2d-e8fea33afaf2"  begin
            include("../ext/ExtendableSparsePardisoExt.jl")
        end
        @require IncompleteLU = "40713840-3770-5561-ab4c-a76e7d0d7895"  begin
            include("../ext/ExtendableSparseIncompleteLUExt.jl")
        end
        @require AlgebraicMultigrid = "2169fc97-5a83-5252-b627-83903c6c433c" begin
            include("../ext/ExtendableSparseAlgebraicMultigridExt.jl")
        end
        @require AMGCLWrap = "4f76b812-4ba5-496d-b042-d70715554288" begin
            include("../ext/ExtendableSparseAMGCLWrapExt.jl")
        end
    end
end

"""
```
ILUTPreconditioner(;droptol=1.0e-3)
ILUTPreconditioner(matrix; droptol=1.0e-3)
```

Create the [`ILUTPreconditioner`](@ref) wrapping the one 
from [IncompleteLU.jl](https://github.com/haampie/IncompleteLU.jl)
For using this, you need to issue `using IncompleteLU`.
"""
function ILUTPreconditioner end
export ILUTPreconditioner

"""
```
AMGPreconditioner(;max_levels=10, max_coarse=10)
AMGPreconditioner(matrix;max_levels=10, max_coarse=10)
```

Create the  [`AMGPreconditioner`](@ref) wrapping the Ruge-Stüben AMG preconditioner from [AlgebraicMultigrid.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl)

!!! warning
     Deprecated in favor of [`RS_AMGPreconditioner`](@ref)

"""
function AMGPreconditioner end 
export AMGPreconditioner

@deprecate AMGPreconditioner() RS_AMGPreconditioner()
@deprecate AMGPreconditioner(A) RS_AMGPreconditioner(A)

"""
```
RS_AMGPreconditioner(;kwargs...)
RS_AMGPreconditioner(matrix;kwargs...)
```

Create the  [`RS_AMGPreconditioner`](@ref) wrapping the Ruge-Stüben AMG preconditioner from [AlgebraicMultigrid.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl)
For `kwargs` see there.
"""
function RS_AMGPreconditioner end 
export RS_AMGPreconditioner


"""
```
SA_AMGPreconditioner(;kwargs...)
SA_AMGPreconditioner(matrix;kwargs...)
```

Create the  [`SA_AMGPreconditioner`](@ref) wrapping the smoothed aggregation AMG preconditioner from [AlgebraicMultigrid.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl)
For `kwargs` see there.
"""
function SA_AMGPreconditioner end 
export SA_AMGPreconditioner




"""
```
AMGCL_AMGPreconditioner(;kwargs...)
AMGCL_AMGPreconditioner(matrix;kwargs...)
```

Create the  [`AMGCL_AMGPreconditioner`](@ref) wrapping AMG preconditioner from [AMGCLWrap.jl](https://github.com/j-fu/AMGCLWrap.jl)
For `kwargs` see there.
"""
function AMGCL_AMGPreconditioner end
export AMGCL_AMGPreconditioner


"""
```
AMGCL_RLXPreconditioner(;kwargs...)
AMGCL_RLXPreconditioner(matrix;kwargs...)
```

Create the  [`AMGCL_RLXPreconditioner`](@ref) wrapping RLX preconditioner from [AMGCLWrap.jl](https://github.com/j-fu/AMGCLWrap.jl)
"""
function AMGCL_RLXPreconditioner end
export AMGCL_RLXPreconditioner




"""
```
PardisoLU(;iparm::Vector, 
           dparm::Vector, 
           mtype::Int)

PardisoLU(matrix; iparm,dparm,mtype)
```

LU factorization based on pardiso. For using this, you need to issue `using Pardiso`
and have the pardiso library from  [pardiso-project.org](https://pardiso-project.org) 
[installed](https://github.com/JuliaSparse/Pardiso.jl#pardiso-60).

The optional keyword arguments `mtype`, `iparm`  and `dparm` are 
[Pardiso internal parameters](https://github.com/JuliaSparse/Pardiso.jl#readme).

Forsetting them, one can also access the `PardisoSolver` e.g. like
```
using Pardiso
plu=PardisoLU()
Pardiso.set_iparm!(plu.ps,5,13.0)
```
"""
function PardisoLU end
export PardisoLU


"""
```
MKLPardisoLU(;iparm::Vector, mtype::Int)

MKLPardisoLU(matrix; iparm, mtype)
```

LU factorization based on pardiso. For using this, you need to issue `using Pardiso`.
This version  uses the early 2000's fork in Intel's MKL library.

The optional keyword arguments `mtype` and `iparm`  are  
[Pardiso internal parameters](https://github.com/JuliaSparse/Pardiso.jl#readme).

For setting them you can also access the `PardisoSolver` e.g. like
```
using Pardiso
plu=MKLPardisoLU()
Pardiso.set_iparm!(plu.ps,5,13.0)
```
"""
function MKLPardisoLU end
export MKLPardisoLU

end # module
