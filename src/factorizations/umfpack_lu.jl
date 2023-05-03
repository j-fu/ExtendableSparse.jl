mutable struct LUFactorization <: AbstractLUFactorization
    A::Union{Nothing, ExtendableSparseMatrix}
    factorization::Union{Nothing, SuiteSparse.UMFPACK.UmfpackLU}
    phash::UInt64
end

LUFactorization() = LUFactorization(nothing,nothing,0)
"""
```
LUFactorization()
LUFactorization(matrix)
```
        
Default LU Factorization. Maps to Sparspak.jl for non-GPL builds, otherwise to UMFPACK.
"""
function LUFactorization end

function update!(lufact::LUFactorization)
    flush!(lufact.A)
    if lufact.A.phash != lufact.phash
        lufact.factorization = LinearAlgebra.lu(lufact.A.cscmatrix)
        lufact.phash = lufact.A.phash
    else
        lufact.factorization = lu!(lufact.factorization, lufact.A.cscmatrix)
    end
    lufact
end

