mutable struct ILUZeroPreconditioner <: AbstractPreconditioner
    A::ExtendableSparseMatrix
    factorization::ILUZero.ILU0Precon
    phash::UInt64
    function ILUZeroPreconditioner()
        p = new()
        p.phash = 0
        p
    end
end

"""
```
ILUZeroPreconditioner()
ILUZeroPreconditioner(matrix)
```
Incomplete LU preconditioner with zero fill-in using  [ILUZero.jl](https://github.com/mcovalt/ILUZero.jl). This preconditioner
also calculates and stores updates to the off-diagonal entries and thus delivers better convergence than  the [`ILU0Preconditioner`](@ref).
"""
function ILUZeroPreconditioner end

function update!(p::ILUZeroPreconditioner)
    flush!(p.A)
    if p.A.phash != p.phash
        p.factorization = ILUZero.ilu0(p.A.cscmatrix)
        p.phash=p.A.phash
    else
        ILUZero.ilu0!(p.factorization, p.A.cscmatrix)
    end
    p
end

allow_views(::ILUZeroPreconditioner)=true
allow_views(::Type{ILUZeroPreconditioner})=true
