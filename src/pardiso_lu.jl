"""
$(TYPEDEF)

LU Factorization
"""
mutable struct PardisoLU{Tv, Ti} <: AbstractExtendableSparseLU{Tv,Ti}
    A::ExtendableSparseMatrix{Tv,Ti}
    ps::Pardiso.AbstractPardisoSolver
    phash::UInt64
end


function Pardiso.set_matrixtype!(ps, A::ExtendableSparseMatrix)
    Acsc=A.cscmatrix
    
    if eltype(Acsc)==Float64 && issymmetric(Acsc)
        Pardiso.set_matrixtype!(ps, Pardiso.REAL_SYM)
    elseif eltype(Acsc)==Float64
        Pardiso.set_matrixtype!(ps, Pardiso.REAL_NONSYM)
    elseif eltype(Acsc)==Complex64 && ishermitian(Acsc)
        Pardiso.set_matrixtype!(ps, Pardiso.COMPLEX_HERM_INDEF)
    elseif eltype(Acsc)==Complex64
        Pardiso.set_matrixtype!(ps, Pardiso.COMPLEX_NONYSYM)
    else
        error("unable to detect matrix type")
    end
end

"""
```
PardisoLU(A; ps=Pardiso.MKLPardisoSolver)
```
"""
function PardisoLU(A::ExtendableSparseMatrix{Tv,Ti};ps::Pardiso.AbstractPardisoSolver=Pardiso.MKLPardisoSolver()) where {Tv,Ti}
    @inbounds flush!(A)
    Acsc=A.cscmatrix
    Pardiso.pardisoinit(ps)
    Pardiso.set_matrixtype!(ps,A)
    Pardiso.fix_iparm!(ps, :N)
    Pardiso.set_phase!(ps, Pardiso.ANALYSIS_NUM_FACT)
    Pardiso.pardiso(ps, Tv[], Acsc, Tv[])
    PardisoLU(A,ps,A.phash)
end

function update!(lufact::PardisoLU{Tv,Ti}) where {Tv, Ti}
    ps=lufact.ps
    flush!(lufact.A)
    Acsc=lufact.A.cscmatrix
    if lufact.phash!=lufact.A.phash
        Pardiso.set_phase!(ps, Pardiso.RELEASE_ALL)
        Pardiso.pardiso(ps, Tv[], Acsc, Tv[])
        Pardiso.set_matrixtype!(ps,lufact.A)
        Pardiso.set_phase!(ps, Pardiso.ANALYSIS_NUM_FACT)
        lufact.phash=lufact.A.phash
    else
        Pardiso.set_phase!(ps, Pardiso.NUM_FACT)
    end
    Pardiso.pardiso(ps, Tv[], Acsc, Tv[])
    lufact
end

function LinearAlgebra.ldiv!(u::AbstractArray{T,1} where T, lufact::PardisoLU, v::AbstractArray{T,1} where T)
    ps=lufact.ps
    Acsc=lufact.A.cscmatrix
    Pardiso.set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    Pardiso.pardiso(ps, u, Acsc, v)
    u
end

LinearAlgebra.ldiv!(fact::PardisoLU, v::AbstractArray{T,1} where T)=ldiv!(v,fact,copy(v))
