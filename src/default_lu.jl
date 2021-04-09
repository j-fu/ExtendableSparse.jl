"""
$(TYPEDEF)

LU Factorization
"""
mutable struct ExtendableSparseLU{Tv, Ti} <: AbstractExtendablePreconditioner{Tv,Ti}
    extmatrix::ExtendableSparseMatrix{Tv,Ti}
    lu
    phash::UInt64
end


"""
$(SIGNATURES)

[`flush!`](@ref) and return LU factorization
"""
function LinearAlgebra.lu(ext::ExtendableSparseMatrix)
    @inbounds flush!(ext)
    ExtendableSparseLU(ext,lu(ext.cscmatrix),ext.phash)
end

"""
$(SIGNATURES)

[`flush!`](@ref) and update LU factorization
"""
function LinearAlgebra.lu!(extlu::ExtendableSparseLU, ext::ExtendableSparseMatrix)
    @inbounds flush!(ext)
    extlu.extmatrix=ext
    update!(extlu)
end

"""
$(SIGNATURES)

[`flush!`](@ref) and update LU factorization for matrix stored within LU factorization.
If necessary, update pattern.
"""
function update!(extlu::ExtendableSparseLU)
    extmatrix=extlu.extmatrix
    flush!(extmatrix)
    if need_symbolic_update(extlu)
        extlu.lu=lu(extmatrix.cscmatrix)
    else
        extlu.lu=lu!(extlu.lu,extmatrix.cscmatrix)
    end
    extlu
end


function LinearAlgebra.:\(extlu::ExtendableSparseLU,B::AbstractVecOrMat{T} where T)
    extlu.lu\B
end

function LinearAlgebra.ldiv!(u::AbstractArray{T,1} where T, extlu::ExtendableSparseLU, v::AbstractArray{T,1} where T)
    ldiv!(u, extlu.lu, v)
end

function LinearAlgebra.ldiv!(extlu::ExtendableSparseLU, v::AbstractArray{T,1} where T)
    u=copy(v)
    ldiv!(u, extlu, v)
    v.=u
end
