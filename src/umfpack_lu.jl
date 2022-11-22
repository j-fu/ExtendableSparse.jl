mutable struct LUFactorization{Tv,Ti} <: AbstractLUFactorization{Tv,Ti}
    A::Union{Nothing,ExtendableSparseMatrix{Tv,Ti}}
    fact::Union{Nothing,SuiteSparse.UMFPACK.UmfpackLU{Tv,Ti}}
    phash::UInt64
end


"""
```
LUFactorization(;valuetype=Float64, indextype=Int64)
LUFactorization(matrix)
```
        
Default Julia LU Factorization based on umfpack.
"""
LUFactorization(;valuetype::Type=Float64,indextype::Type=Int64)=LUFactorization{valuetype,indextype}(nothing,nothing,0)


function update!(lufact::LUFactorization)
    flush!(lufact.A)
    if lufact.A.phash!=lufact.phash
        lufact.fact=lu(lufact.A.cscmatrix)
        lufact.phash=lufact.A.phash
    else
        lufact.fact=lu!(lufact.fact,lufact.A.cscmatrix)
    end
    lufact
end

LinearAlgebra.ldiv!(fact::LUFactorization, v)=fact.fact\v
LinearAlgebra.ldiv!(u,fact::LUFactorization, v)=u.=fact.fact\v

