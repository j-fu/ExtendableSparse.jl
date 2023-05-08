module ExtendableSparseIncompleteLUExt
using ExtendableSparse
using ..IncompleteLU

import ExtendableSparse: @makefrommatrix, AbstractPreconditioner, update!

mutable struct ILUTPreconditioner <: AbstractPreconditioner
    A::ExtendableSparseMatrix
    factorization::IncompleteLU.ILUFactorization
    droptol::Float64
    function ILUTPreconditioner(; droptol = 1.0e-3)
        p = new()
        p.droptol = droptol
        p
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

@eval begin
    @makefrommatrix ILUTPreconditioner
end

function update!(precon::ILUTPreconditioner)
    A = precon.A
    @inbounds flush!(A)
    precon.factorization = IncompleteLU.ilu(A.cscmatrix; τ = precon.droptol)
end
end

