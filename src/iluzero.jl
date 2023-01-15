mutable struct ILUZeroPreconditioner{Tv, Ti} <: AbstractPreconditioner{Tv, Ti}
    A::ExtendableSparseMatrix{Tv, Ti}
    ilu::ILUZero.ILU0Precon{Tv, Ti, Tv}
    phash::UInt64
    function ILUZeroPreconditioner{Tv, Ti}() where {Tv, Ti}
        p = new()
        p.phash = 0
        p
    end
end

"""
```
ILUZeroPreconditioner(;valuetype=Float64,indextype=Int64)
ILUZeroPreconditioner(matrix)
```

Incomplete LU preconditioner with zero fill-in using  [ILUZero.jl](https://github.com/mcovalt/ILUZero.jl). This preconditioner
also calculates and stores updates to the off-diagonal entries and thus delivers better convergence than  the [`ILU0Preconditoner`](@ref).
"""
function ILUZeroPreconditioner(; valuetype::Type = Float64, indextype::Type = Int64)
    ILUZeroPreconditioner{valuetype, indextype}()
end

function update!(precon::ILUZeroPreconditioner{Tv, Ti}) where {Tv, Ti}
    flush!(precon.A)
    if precon.A.phash != precon.phash
        precon.ilu = ilu0(precon.A.cscmatrix)
    else
        ilu0!(precon.ilu, precon.A.cscmatrix)
    end
    precon
end

LinearAlgebra.ldiv!(u, precon::ILUZeroPreconditioner, v) = ldiv!(u, precon.ilu, v)
LinearAlgebra.ldiv!(precon::ILUZeroPreconditioner, v) = ldiv!(precon.ilu, v)
