abstract type AbstractPardisoLU{Tv,Ti} <: AbstractLUFactorization{Tv,Ti} end


mutable struct PardisoLU{Tv, Ti} <: AbstractPardisoLU{Tv,Ti}
    A::Union{ExtendableSparseMatrix{Tv,Ti},Nothing}
    ps::Pardiso.AbstractPardisoSolver
    phash::UInt64
end

"""
$(SIGNATURES)

LU factorization based on pardiso. For using this, you need to issue `using Pardiso`
and have the pardiso library from  [pardiso-project.org](https://pardiso-project.org) 
installed.
"""
PardisoLU()=PardisoLU{Float64,Int64}(nothing,Pardiso.PardisoSolver(),0)
PardisoLU{Tv,Ti}() where {Tv,Ti} =PardisoLU{Tv,Ti}(nothing,Pardiso.PardisoSolver(),0)
PardisoLU(A::ExtendableSparseMatrix{Tv,Ti}) where {Tv,Ti} = factorize!(PardisoLU(),A)


mutable struct MKLPardisoLU{Tv, Ti} <: AbstractPardisoLU{Tv,Ti}
    A::Union{ExtendableSparseMatrix{Tv,Ti},Nothing}
    ps::Pardiso.AbstractPardisoSolver
    phash::UInt64
end

"""
$(SIGNATURES)

LU factorization based on pardiso. For using this, you need to issue `using Pardiso`.
This version  uses the early 2000's fork in Intel's MKL library.
"""
MKLPardisoLU()=MKLPardisoLU{Float64,Int64}(nothing,Pardiso.MKLPardisoSolver(),0)
MKLPardisoLU(A::ExtendableSparseMatrix{Tv,Ti}) where {Tv,Ti}=factorize!(MKLPardisoLU(),A)
MKLPardisoLU{Tv,Ti}() where {Tv,Ti} =PardisoLU{Tv,Ti}(nothing,Pardiso.MKLPardisoSolver(),0)


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


function update!(lufact::AbstractPardisoLU{Tv,Ti}) where {Tv, Ti}
    ps=lufact.ps
    flush!(lufact.A)
    Acsc=lufact.A.cscmatrix
    if lufact.phash!=lufact.A.phash
        Pardiso.pardisoinit(ps)
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

function LinearAlgebra.ldiv!(u::AbstractArray{T,1} where T, lufact::AbstractPardisoLU, v::AbstractArray{T,1} where T)
    ps=lufact.ps
    Acsc=lufact.A.cscmatrix
    Pardiso.set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    Pardiso.pardiso(ps, u, Acsc, v)
    u
end

LinearAlgebra.ldiv!(fact::AbstractPardisoLU, v::AbstractArray{T,1} where T)=ldiv!(v,fact,copy(v))

