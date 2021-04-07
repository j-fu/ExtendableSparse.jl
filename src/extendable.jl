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
function ExtendableSparseMatrix{Tv,Ti}(m, n) where{Tv,Ti<:Integer}
    ExtendableSparseMatrix{Tv,Ti}(spzeros(Tv,Ti,m,n),nothing,time())
end

"""
$(SIGNATURES)

Create empty ExtendableSparseMatrix.
"""
function ExtendableSparseMatrix(valuetype::Type{Tv},indextype::Type{Ti},m, n) where{Tv,Ti<:Integer}
    ExtendableSparseMatrix{Tv,Ti}(m,n)
end

"""
$(SIGNATURES)

Create empty ExtendablSparseMatrix.
This is a pendant to spzeros.
"""
ExtendableSparseMatrix(valuetype::Type{Tv},m, n) where{Tv}=ExtendableSparseMatrix{Tv,Int}(m,n)


"""
$(SIGNATURES)

Create empty ExtendableSparseMatrix.
This is a pendant to spzeros.
"""
ExtendableSparseMatrix(m, n)=ExtendableSparseMatrix{Float64,Int}(m,n)


"""
$(SIGNATURES)

  Create ExtendableSparseMatrix from SparseMatrixCSC
"""
function ExtendableSparseMatrix(csc::SparseMatrixCSC{Tv,Ti}) where{Tv,Ti<:Integer}
    return ExtendableSparseMatrix{Tv,Ti}(csc, nothing, time())
end

"""
$(SIGNATURES)
  Create similar extendableSparseMatrix
"""
Base.similar(m::ExtendableSparseMatrix{Tv,Ti}) where {Tv,Ti}=ExtendableSparseMatrix{Tv,Ti}(size(m)...)


"""
$(SIGNATURES)

Update element of the matrix  with operation `op`.
This can replace the following code and save one index
search during acces:

```@example
using ExtendableSparse # hide
A=ExtendableSparseMatrix(3,3)
A[1,2]+=0.1
A
```

```@example
using ExtendableSparse # hide

A=ExtendableSparseMatrix(3,3)
updateindex!(A,+,0.1,1,2)
A
```

If `v` is zero, no new entry is created.
"""
function updateindex!(ext::ExtendableSparseMatrix{Tv,Ti}, op,v, i,j) where {Tv,Ti<:Integer}
    k=findindex(ext.cscmatrix,i,j)
    if k>0
        ext.cscmatrix.nzval[k]=op(ext.cscmatrix.nzval[k],v)
    else
        if ext.lnkmatrix==nothing
            ext.lnkmatrix=SparseMatrixLNK{Tv, Ti}(ext.cscmatrix.m, ext.cscmatrix.n)
        end
        updateindex!(ext.lnkmatrix,op,v,i,j)
    end
    ext
end


"""
$(SIGNATURES)
Like [`updateindex!`](@ref) but without 
checking if v is zero.

"""
function rawupdateindex!(ext::ExtendableSparseMatrix{Tv,Ti}, op,v, i,j) where {Tv,Ti<:Integer}
    k=findindex(ext.cscmatrix,i,j)
    if k>0
        ext.cscmatrix.nzval[k]=op(ext.cscmatrix.nzval[k],v)
    else
        if ext.lnkmatrix==nothing
            ext.lnkmatrix=SparseMatrixLNK{Tv, Ti}(ext.cscmatrix.m, ext.cscmatrix.n)
        end
        rawupdateindex!(ext.lnkmatrix,op,v,i,j)
    end
    ext
end


"""
$(SIGNATURES)

Find index in CSC matrix and set value if it exists. Otherwise,
set index in extension if `v` is nonzero.
"""

function Base.setindex!(ext::ExtendableSparseMatrix{Tv,Ti}, v, i,j) where {Tv,Ti}
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
function Base.getindex(ext::ExtendableSparseMatrix{Tv,Ti},i, j)  where{Tv,Ti<:Integer}
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

Show matrix
"""
function Base.show(io::IO,::MIME"text/plain",ext::ExtendableSparseMatrix)
    flush!(ext)
    xnnz = nnz(ext)
    m, n = size(ext)
    print(io, m, "×", n, " ", typeof(ext), " with ", xnnz, " stored ",
          xnnz == 1 ? "entry" : "entries")
    if !(m == 0 || n == 0 || xnnz==0)
        print(io, ":")
        show(IOContext(io), ext.cscmatrix)
    end
end



"""
$(SIGNATURES)

If there are new entries in extension, create new CSC matrix by adding the
cscmatrix and linked list matrix and reset the linked list based extension.
"""
function flush!(ext::ExtendableSparseMatrix)
    if ext.lnkmatrix!=nothing && nnz(ext.lnkmatrix)>0
        ext.cscmatrix=ext.lnkmatrix+ext.cscmatrix
        ext.lnkmatrix=nothing
        ext.pattern_timestamp=time()
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

[`\\`](@ref) for extmatrix
"""
function LinearAlgebra.:\(ext::ExtendableSparseMatrix,B::AbstractVecOrMat{T} where T)
    flush!(ext)
    ext.cscmatrix\B
end


"""
$(SIGNATURES)

[`\\`](@ref) for Symmetric{ExtendableSparse}
"""
function LinearAlgebra.:\(symm_ext::Symmetric{Tm, ExtendableSparseMatrix{Tm, Ti}}, B::AbstractVecOrMat{T} where T) where{Tm,Ti}
    flush!(symm_ext.data)
    symm_csc=Symmetric(symm_ext.data.cscmatrix,Symbol(symm_ext.uplo))
    symm_csc\B
end


"""
$(SIGNATURES)

[`\\`](@ref) for Hermitian{ExtendableSparse}
"""
function LinearAlgebra.:\(symm_ext::Hermitian{Tm, ExtendableSparseMatrix{Tm, Ti}}, B::AbstractVecOrMat{T} where T) where{Tm,Ti}
    flush!(symm_ext.data)
    symm_csc=Hermitian(symm_ext.data.cscmatrix,Symbol(symm_ext.uplo))
    symm_csc\B
end


"""
$(SIGNATURES)

[`flush!`](@ref) and ldiv with ext.cscmatrix
"""
function  LinearAlgebra.ldiv!(r, ext::ExtendableSparse.ExtendableSparseMatrix, x)
    @inbounds flush!(ext)
    return LinearAlgebra.ldiv!(r,ext.cscmatrix,x)
end

"""
$(SIGNATURES)

[`flush!`](@ref) and multiply with ext.cscmatrix
"""
function  LinearAlgebra.mul!(r,ext::ExtendableSparse.ExtendableSparseMatrix, x)
    @inbounds flush!(ext)
    return LinearAlgebra.mul!(r,ext.cscmatrix,x)
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
function Base.:+(ext::ExtendableSparseMatrix,csc::SparseMatrixCSC)
    @inbounds flush!(ext)
    return ext.cscmatrix+csc
end

"""
$(SIGNATURES)

Subtract  SparseMatrixCSC matrix from  [`ExtendableSparseMatrix`](@ref)  ext.
"""
function Base.:-(ext::ExtendableSparseMatrix,csc::SparseMatrixCSC)
    @inbounds flush!(ext)
    return ext.cscmatrix-csc
end

"""
$(SIGNATURES)

Subtract  [`ExtendableSparseMatrix`](@ref)  ext from  SparseMatrixCSC.
"""
function Base.:-(csc::SparseMatrixCSC,ext::ExtendableSparseMatrix)
    @inbounds flush!(ext)
    return csc - ext.cscmatrix
end


"""
$(SIGNATURES)
"""
function SparseArrays.dropzeros!(ext::ExtendableSparseMatrix)
    @inbounds flush!(ext)
    dropzeros!(ext.cscmatrix)
end
