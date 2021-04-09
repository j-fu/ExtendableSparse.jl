using Pardiso

"""
$(TYPEDEF)

LU Factorization
"""
mutable struct PardisoLU{Tv, Ti} <: AbstractExtendablePreconditioner{Tv,Ti}
    extmatrix::ExtendableSparseMatrix{Tv,Ti}
    ps::Pardiso.AbstractPardisoSolver
    phash::UInt64
end


function PardisoLU(ext::ExtendableSparseMatrix;ps::Pardiso.AbstractPardisoSolver=PardisoSolver())
    @inbounds flush!(ext)
    A=ext.cscmatrix
    eltype(A) == Float64 ? set_matrixtype!(ps, Pardiso.REAL_NONSYM) : set_matrixtype!(ps,  Pardiso.COMPLEX_NONSYM)
    pardisoinit(ps)
    fix_iparm!(ps, :N)
    set_iparm!(ps,2,3)
    Tv=Float64
    set_phase!(ps, Pardiso.ANALYSIS_NUM_FACT)
    pardiso(ps, Tv[], A, Tv[])
    PardisoLU(ext,ps,ext.phash)
end


"""
$(SIGNATURES)

[`flush!`](@ref) and update LU factorization for matrix stored within LU factorization.
If necessary, update pattern.
"""
function update!(extlu::PardisoLU)
    ps=extlu.ps
    Tv=Float64
    flush!(extlu.extmatrix)
    A=extlu.extmatrix.cscmatrix
    if need_symbolic_update(extlu)
        set_phase!(ps, Pardiso.RELEASE_ALL)
        pardiso(ps, Tv[], A, Tv[])
        set_phase!(ps, Pardiso.ANALYSIS_NUM_FACT)
    else
        set_phase!(ps, Pardiso.NUM_FACT)
    end
    pardiso(ps, Tv[], A, Tv[])
    extlu
end

function LinearAlgebra.ldiv!(u::AbstractArray{T,1} where T, extlu::PardisoLU, v::AbstractArray{T,1} where T)
    ps=extlu.ps
    A=extlu.extmatrix.cscmatrix
    set_iparm!(ps,8,0)
    set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    pardiso(ps, u, A, v)
end


"""
$(SIGNATURES)

[`flush!`](@ref) and update LU factorization
"""
function LinearAlgebra.lu!(extlu::PardisoLU, ext::ExtendableSparseMatrix)
    extlu.extmatrix=ext
    update!(extlu)
end

LinearAlgebra.ldiv!(extlu::PardisoLU, v::AbstractArray{T,1} where T)=ldiv!(v, extlu, copy(v))
