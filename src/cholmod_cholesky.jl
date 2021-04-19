"""
$(TYPEDEF)

Default Julia LU Factorization based on umfpack.
"""
mutable struct ExtendableSparseCholmodCholesky{Tv, Ti} <: AbstractExtendableSparseLU{Tv,Ti}
    A::ExtendableSparseMatrix{Tv,Ti}
    A64
    fact::SuiteSparse.CHOLMOD.Factor{Tv}
    phash::UInt64
end

"""
```
ExtendableSparseCholmodCholesky(A)
```
"""
function ExtendableSparseCholmodCholesky(A::ExtendableSparseMatrix{Tv,Ti}) where {Tv,Ti}
    flush!(A)
    A64=Symmetric(SparseMatrixCSC{Float64,Int64}(A.cscmatrix))
    ExtendableSparseCholmodCholesky(A,A64,cholesky(A64),A.phash)
end

function update!(cholfact::ExtendableSparseCholmodCholesky)
    flush!(cholfact.A)
    if cholfact.A.phash!=cholfact.phash
        cholfact.A64=Symmetric{SparseMatrixCSC{Float64,Int64}(A.cscmatrix)}
        cholfact.fact=cholesky(cholfact.fact,cholfact.A64)
        cholfact.phash=cholfact.A.phash
    else
        cholfact.A64.data.nzval.=cholfact.A.cscmatrix.nzval
        cholfact.fact=cholesky!(cholfact.fact,cholfact.A64)
    end
    cholfact
end


LinearAlgebra.ldiv!(fact::ExtendableSparseCholmodCholesky, v)=fact.fact\v
LinearAlgebra.ldiv!(u,fact::ExtendableSparseCholmodCholesky, v)=u.=fact.fact\v

