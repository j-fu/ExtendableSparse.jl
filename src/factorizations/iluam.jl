mutable struct ILUAMPreconditioner <: AbstractPreconditioner
    A::ExtendableSparseMatrix
    factorization::ILUAM.ILUAMPrecon
    phash::UInt64
    function ILUAMPreconditioner()
        p = new()
        p.phash = 0
        p
    end
end

"""
```
ILUAMPreconditioner()
ILUAMPreconditioner(matrix)
```
Incomplete LU preconditioner with zero fill-in using ... . This preconditioner
also calculates and stores updates to the off-diagonal entries and thus delivers better convergence than  the [`ILU0Preconditioner`](@ref).
"""
function ILUAMPreconditioner end

function update!(p::ILUAMPreconditioner)
    flush!(p.A)
    if p.A.phash != p.phash
        p.factorization = ILUAM.iluAM(p.A.cscmatrix)
        p.phash=p.A.phash
    else
        ILUAM.ilu0!(p.factorization, p.A.cscmatrix)
    end
    p
end

allow_views(::ILUAMPreconditioner)=true
allow_views(::Type{ILUAMPreconditioner})=true

