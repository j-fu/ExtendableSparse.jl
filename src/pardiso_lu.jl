"""
$(TYPEDEF)

LU Factorization
"""
mutable struct PardisoLU{Tv, Ti} <: AbstractExtendableSparseLU{Tv,Ti}
    A::ExtendableSparseMatrix{Tv,Ti}
    ps::Pardiso.AbstractPardisoSolver
    phash::UInt64
end


function PardisoLU(A::ExtendableSparseMatrix{Tv,Ti};ps::Pardiso.AbstractPardisoSolver=Pardiso.MKLPardisoSolver()) where {Tv,Ti}
    @inbounds flush!(A)
    Acsc=A.cscmatrix
    eltype(Acsc) == Float64 ? Pardiso.set_matrixtype!(ps, Pardiso.REAL_NONSYM) : Pardiso.set_matrixtype!(ps,  Pardiso.COMPLEX_NONSYM)
    Pardiso.pardisoinit(ps)
    Pardiso.fix_iparm!(ps, :N)
    Pardiso.set_phase!(ps, Pardiso.ANALYSIS_NUM_FACT)
    Pardiso.pardiso(ps, Tv[], Acsc, Tv[])
    PardisoLU(A,ps,A.phash)
end


"""
$(SIGNATURES)

[`flush!`](@ref) and update LU factorization for matrix stored within LU factorization.
If necessary, update pattern.
"""
function factorize!(lufact::PardisoLU, A::ExtendableSparseMatrix; kwargs...)
    ps=lufact.ps
    Tv=Float64
    flush!(A)
    lufact.A=A
    Acsc=lufact.A.cscmatrix
    if lufact.phash!=A.phash
        Pardiso.set_phase!(ps, Pardiso.RELEASE_ALL)
        Pardiso.pardiso(ps, Tv[], Acsc, Tv[])
        Pardiso.set_phase!(ps, Pardiso.ANALYSIS_NUM_FACT)
    else
        Pardiso.set_phase!(ps, Pardiso.NUM_FACT)
    end
    Pardiso.pardiso(ps, Tv[], Acsc, Tv[])
    lufact
end

update!(lufact::PardisoLU)=factorize!(lufact,lufact.A)


function LinearAlgebra.ldiv!(u::AbstractArray{T,1} where T, lufact::PardisoLU, v::AbstractArray{T,1} where T)
    ps=lufact.ps
    Acsc=lufact.A.cscmatrix
    Pardiso.set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    Pardiso.pardiso(ps, u, Acsc, v)
    u
end


LinearAlgebra.ldiv!(lufact::PardisoLU, v::AbstractArray{T,1} where T)=ldiv!(v, lufact, copy(v))


