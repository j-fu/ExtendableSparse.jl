mutable struct CholeskyFactorization <: AbstractLUFactorization
    A::Union{ExtendableSparseMatrix, Nothing}
    fact::Union{SuiteSparse.CHOLMOD.Factor, Nothing}
    phash::UInt64
    A64::Any
end

"""
CholeskyFactorization(;valuetype=Float64, indextype=Int64)
CholeskyFactorization(matrix)

Default Cholesky factorization via cholmod.
"""
function CholeskyFactorization()
    CholeskyFactorization(nothing, nothing, 0, nothing)
end

function update!(cholfact::CholeskyFactorization)
    A = cholfact.A
    flush!(A)
    if A.phash != cholfact.phash
        cholfact.A64 = Symmetric(A.cscmatrix)
        cholfact.fact = cholesky(cholfact.A64)
        cholfact.phash = A.phash
    else
        cholfact.A64.data.nzval .= A.cscmatrix.nzval
        cholfact.fact = cholesky!(cholfact.fact, cholfact.A64)
    end
    cholfact
end

LinearAlgebra.ldiv!(fact::CholeskyFactorization, v) = fact.fact \ v
LinearAlgebra.ldiv!(u, fact::CholeskyFactorization, v) = u .= fact.fact \ v
