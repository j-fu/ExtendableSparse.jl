##################################################################
"""
$(TYPEDEF)

Extendable sparse matrix. A nonzero  entry of this matrix is contained
either in cscmatrix, or in lnkmatrix, never in both.

$(TYPEDFIELDS)
"""
mutable struct ExtendableSparseMatrix{Tv,Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti}
    """
    Final matrix data
    """
    cscmatrix::SparseMatrixCSC{Tv,Ti}

    """
    Linked list structure holding data of extension
    """
    lnkmatrix::Union{SparseMatrixLNK{Tv,Ti},Nothing}

    """
    Time stamp of last pattern update
    """
    pattern_timestamp::Float64
end


"""
$(SIGNATURES)

Create empty ExtendableSparseMatrix.
"""
function ExtendableSparseMatrix{Tv,Ti}(m::Integer, n::Integer) where{Tv,Ti<:Integer}
    ExtendableSparseMatrix{Tv,Ti}(spzeros(Tv,Ti,m,n),nothing,time())
end

"""
$(SIGNATURES)

Create empty ExtendableSparseMatrix.
"""
function ExtendableSparseMatrix(valuetype::Type{Tv},indextype::Type{Ti},m::Integer, n::Integer) where{Tv,Ti<:Integer}
    ExtendableSparseMatrix{Tv,Ti}(m,n)
end

"""
$(SIGNATURES)

Create empty ExtendablSparseMatrix.
This is a pendant to spzeros.
"""
ExtendableSparseMatrix(valuetype::Type{Tv},m::Integer, n::Integer) where{Tv}=ExtendableSparseMatrix{Tv,Int}(m,n)


"""
$(SIGNATURES)

Create empty ExtendableSparseMatrix.
This is a pendant to spzeros.
"""
ExtendableSparseMatrix(m::Integer, n::Integer)=ExtendableSparseMatrix{Float64,Int}(m,n)


"""
$(SIGNATURES)

  Create ExtendableSparseMatrix from SparseMatrixCSC
"""
function ExtendableSparseMatrix(csc::SparseMatrixCSC{Tv,Ti}) where{Tv,Ti<:Integer}
    return ExtendableSparseMatrix{Tv,Ti}(csc, nothing, time())
end



"""
$(SIGNATURES)

Return index corresponding to entry [i,j] in the array of nonzeros,
if the entry exists, otherwise, return 0.
"""
function findindex(csc::SparseMatrixCSC{T}, i::Integer, j::Integer) where T
    if !(1 <= i <= csc.m && 1 <= j <= csc.n); throw(BoundsError()); end
    r1 = Int(csc.colptr[j])
    r2 = Int(csc.colptr[j+1]-1)
    if r1>r2
        return 0
    end

    # See sparsematrix.jl
    r1 = searchsortedfirst(csc.rowval, i, r1, r2, Base.Forward)
    if (r1>r2 ||csc.rowval[r1] != i)
        return 0
    end
    return r1
end



"""
$(SIGNATURES)

Find index in CSC matrix and set value if it exists. Otherwise,
set index in extension.
"""

function Base.setindex!(ext::ExtendableSparseMatrix{Tv,Ti}, v, i,j) where{Tv,Ti<:Integer}
    k=findindex(ext.cscmatrix,i,j)
    if k>0
        ext.cscmatrix.nzval[k]=v
    else
        if ext.lnkmatrix==nothing
            ext.lnkmatrix=SparseMatrixLNK{Tv, Ti}(ext.cscmatrix.m, ext.cscmatrix.n)
        end
        ext.lnkmatrix[i,j]=v
    end
end



"""
$(SIGNATURES)

Find index in CSC matrix and return value, if it exists.
Otherwise, return value from extension.
"""
function Base.getindex(ext::ExtendableSparseMatrix{Tv,Ti},i::Integer, j::Integer) where{Tv,Ti<:Integer}
    k=findindex(ext.cscmatrix,i,j)
    if k>0
        return ext.cscmatrix.nzval[k]
    elseif ext.lnkmatrix==nothing
        return zero(Tv)
    else
        return ext.lnkmatrix[i,j]
    end
end

"""
$(SIGNATURES)

Size of ExtendableSparseMatrix.
"""
Base.size(ext::ExtendableSparseMatrix) = (ext.cscmatrix.m, ext.cscmatrix.n)



"""
$(SIGNATURES)

If there are new entries in extension, create new CSC matrix by adding the
cscmatrix and linked list matrix and reset the linked list based extension.
"""
function flush!(ext::ExtendableSparseMatrix{Tv,Ti}) where {Tv, Ti<:Integer}
    if ext.lnkmatrix!=nothing && nnz(ext.lnkmatrix)>0
        ext.cscmatrix=ext.lnkmatrix+ext.cscmatrix
        ext.lnkmatrix=nothing
        ext.pattern_timestamp
    end
    return ext
end

"""
$(SIGNATURES)

[`flush!`](@ref) and return number of nonzeros in ext.cscmatrix.
"""
function SparseArrays.nnz(ext::ExtendableSparseMatrix)
    @inbounds flush!(ext)
    return nnz(ext.cscmatrix)
end


"""
$(SIGNATURES)

[`flush!`](@ref) and return nonzeros in ext.cscmatrix.
"""
function SparseArrays.nonzeros(ext::ExtendableSparseMatrix)
    @inbounds flush!(ext)
    return nonzeros(ext.cscmatrix)
end



"""
$(SIGNATURES)

[`flush!`](@ref) and return rowvals in ext.cscmatrix.
"""
function SparseArrays.rowvals(ext::ExtendableSparseMatrix)
    @inbounds flush!(ext)
    return rowvals(ext.cscmatrix)
end


"""
$(SIGNATURES)

[`flush!`](@ref) and return colptr of  in ext.cscmatrix.
"""
function colptrs(ext::ExtendableSparseMatrix)
    @inbounds flush!(ext)
    return ext.cscmatrix.colptr
end


"""
$(SIGNATURES)

[`flush!`](@ref) and return findnz(ext.cscmatrix).
"""
function SparseArrays.findnz(ext::ExtendableSparseMatrix)
    @inbounds flush!(ext)
    return findnz(ext.cscmatrix)
end


"""
$(SIGNATURES)

[`flush!`](@ref) and return LU factorization of ext.cscmatrix
"""
function LinearAlgebra.lu(ext::ExtendableSparseMatrix)
    @inbounds flush!(ext)
    return LinearAlgebra.lu(ext.cscmatrix)
end

"""
$(SIGNATURES)

[`flush!`](@ref) and multiply with ext.cscmatrix
"""
function  LinearAlgebra.mul!(r::AbstractArray{T,1} where T, ext::ExtendableSparse.ExtendableSparseMatrix, x::AbstractArray{T,1} where T)
    @inbounds flush!(ext)
    return LinearAlgebra.mul!(r,ext.cscmatrix,x)
end


"""
$(SIGNATURES)

[`flush!`](@ref) and ldiv with ext.cscmatrix
"""
function  LinearAlgebra.ldiv!(r::AbstractArray{T,1} where T, ext::ExtendableSparse.ExtendableSparseMatrix, x::AbstractArray{T,1} where T)
    @inbounds flush!(ext)
    return LinearAlgebra.ldiv!(r,ext.cscmatrix,x)
end

"""
$(SIGNATURES)

[`flush!`](@ref) and multiply with ext.cscmatrix
"""
function  LinearAlgebra.mul!(r::AbstractArray{T,2} where T, ext::ExtendableSparse.ExtendableSparseMatrix, x::AbstractArray{T,2} where T)
    @inbounds flush!(ext)
    return LinearAlgebra.mul!(r,ext.cscmatrix,x)
end


"""
$(SIGNATURES)

[`flush!`](@ref) and ldiv with ext.cscmatrix
"""
function  LinearAlgebra.ldiv!(r::AbstractArray{T,2} where T, ext::ExtendableSparse.ExtendableSparseMatrix, x::AbstractArray{T,2} where T)
    @inbounds flush!(ext)
    return LinearAlgebra.ldiv!(r,ext.cscmatrix,x)
end



"""
$(SIGNATURES)

[`flush!`](@ref) and calculate norm from cscmatrix
"""
function LinearAlgebra.norm(A::ExtendableSparseMatrix, p::Real=2)
    @time @inbounds flush!(A)
    return LinearAlgebra.norm(A.cscmatrix,p)
end

"""
$(SIGNATURES)

[`flush!`](@ref) and calculate opnorm from cscmatrix
"""
function LinearAlgebra.opnorm(A::ExtendableSparseMatrix, p::Real=2)
    @inbounds flush!(A)
    return LinearAlgebra.opnorm(A.cscmatrix,p)
end

"""
$(SIGNATURES)

[`flush!`](@ref) and calculate cond from cscmatrix
"""
function LinearAlgebra.cond(A::ExtendableSparseMatrix, p::Real=2)
    @inbounds flush!(A)
    return LinearAlgebra.cond(A.cscmatrix,p)
end



"""
$(SIGNATURES)

Add SparseMatrixCSC matrix and [`ExtendableSparseMatrix`](@ref)  ext.
"""
function Base.:+(ext::ExtendableSparseMatrix{Tv,Ti},csc::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti<:Integer}
    @inbounds flush!(ext)
    return ext.cscmatrix+csc
end

"""
$(SIGNATURES)

Subtract  SparseMatrixCSC matrix from  [`ExtendableSparseMatrix`](@ref)  ext.
"""
function Base.:-(ext::ExtendableSparseMatrix{Tv,Ti},csc::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti<:Integer}
    @inbounds flush!(ext)
    return ext.cscmatrix-csc
end

"""
$(SIGNATURES)

Subtract  [`ExtendableSparseMatrix`](@ref)  ext from  SparseMatrixCSC.
"""
function Base.:-(csc::SparseMatrixCSC{Tv,Ti},ext::ExtendableSparseMatrix{Tv,Ti}) where {Tv,Ti<:Integer}
    @inbounds flush!(ext)
    return csc - ext.cscmatrix
end
