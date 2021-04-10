"""
$(TYPEDEF)

Default Julia LU Factorization based on umfpack.
"""
mutable struct ExtendableSparseUmfpackLU{Tv, Ti} <: AbstractExtendableSparseLU{Tv,Ti}
    A::ExtendableSparseMatrix{Tv,Ti}
    fact::SuiteSparse.UMFPACK.UmfpackLU{Tv,Ti}
    phash::UInt64
end

"""
```
ExtendableSparseUmfpackLU(A)
```
"""
function ExtendableSparseUmfpackLU(A::ExtendableSparseMatrix{Tv,Ti}) where {Tv,Ti}
    flush!(A)
    ExtendableSparseUmfpackLU(A,lu(A.cscmatrix),A.phash)
end

function update!(lufact::ExtendableSparseUmfpackLU)
    flush!(lufact.A)
    if lufact.A.phash!=lufact.phash
        lufact.fact=lu(lufact.A.cscmatrix)
        lufact.phash=lufact.A.phash
    else
        lufact.fact=lu!(lufact.fact,lufact.A.cscmatrix)
    end
    lufact
end

