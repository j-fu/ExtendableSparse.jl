"""
$(TYPEDEF)

LU Factorization
"""
mutable struct PardisoLU{Tv, Ti} <: AbstractExtendableSparseLU{Tv,Ti}
    extmatrix::ExtendableSparseMatrix{Tv,Ti}
    ps::Pardiso.AbstractPardisoSolver
    phash::UInt64
end

println( "pardiso!")

function PardisoLU(ext::ExtendableSparseMatrix;ps::Pardiso.AbstractPardisoSolver=Pardiso.MKLPardisoSolver())
    @inbounds flush!(ext)
    A=ext.cscmatrix
    eltype(A) == Float64 ? Pardiso.set_matrixtype!(ps, Pardiso.REAL_NONSYM) : Pardiso.set_matrixtype!(ps,  Pardiso.COMPLEX_NONSYM)
    Pardiso.pardisoinit(ps)
    Pardiso.fix_iparm!(ps, :N)
    Tv=Float64
    Pardiso.set_phase!(ps, Pardiso.ANALYSIS_NUM_FACT)
    Pardiso.pardiso(ps, Tv[], A, Tv[])
    PardisoLU(ext,ps,ext.phash)
end


"""
$(SIGNATURES)

[`flush!`](@ref) and update LU factorization for matrix stored within LU factorization.
If necessary, update pattern.
"""
function factorize!(lufact::PardisoLU, Aext::ExtendableSparseMatrix; kwargs...)
    ps=lufact.ps
    Tv=Float64
    flush!(Aext)
    lufact.extmatrix=Aext
    A=lufact.extmatrix.cscmatrix
    if lufact.phash!=Aext.phash
        Pardiso.set_phase!(ps, Pardiso.RELEASE_ALL)
        Pardiso.pardiso(ps, Tv[], A, Tv[])
        Pardiso.set_phase!(ps, Pardiso.ANALYSIS_NUM_FACT)
    else
        Pardiso.set_phase!(ps, Pardiso.NUM_FACT)
    end
    Pardiso.pardiso(ps, Tv[], A, Tv[])
    lufact
end

function LinearAlgebra.ldiv!(u::AbstractArray{T,1} where T, lufact::PardisoLU, v::AbstractArray{T,1} where T)
    ps=lufact.ps
    A=lufact.extmatrix.cscmatrix
    Pardiso.set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    Pardiso.pardiso(ps, u, A, v)
end


"""
$(SIGNATURES)

[`flush!`](@ref) and update LU factorization
"""
function LinearAlgebra.lu!(lufact::PardisoLU, ext::ExtendableSparseMatrix)
    lufact.extmatrix=ext
    update!(lufact)
end

LinearAlgebra.ldiv!(lufact::PardisoLU, v::AbstractArray{T,1} where T)=ldiv!(v, lufact, copy(v))
