mutable struct PILUAMPreconditioner <: AbstractPreconditioner
    A::ExtendableSparseMatrixParallel
    factorization::PILUAMPrecon
    phash::UInt64
    function PILUAMPreconditioner()
        p = new()
        p.phash = 0
        p
    end
end

"""
```
PILUAMPreconditioner()
PILUAMPreconditioner(matrix)
```
Incomplete LU preconditioner with zero fill-in using ... . This preconditioner
also calculates and stores updates to the off-diagonal entries and thus delivers better convergence than  the [`ILU0Preconditioner`](@ref).
"""
function PILUAMPreconditioner end

function update!(p::PILUAMPreconditioner)
    #@warn "Should flush now", nnz_noflush(p.A)
    flush!(p.A)
    if p.A.phash != p.phash
        p.factorization = piluAM(p.A)
        p.phash=p.A.phash
    else
        piluAM!(p.factorization, p.A)
    end
    p
end

allow_views(::PILUAMPreconditioner)=true
allow_views(::Type{PILUAMPreconditioner})=true

