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
end


"""
$(SIGNATURES)

Create empty ExtendableSparseMatrix.
"""
function ExtendableSparseMatrix{Tv,Ti}(m::Integer, n::Integer) where{Tv,Ti<:Integer}
    ExtendableSparseMatrix{Tv,Ti}(spzeros(Tv,Ti,m,n),nothing)
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
    return ExtendableSparseMatrix{Tv,Ti}(csc, nothing)
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
function Base.setindex!(ext::ExtendableSparseMatrix{Tv,Ti}, v, i::Integer, j::Integer) where{Tv,Ti<:Integer}
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

Number of nonzeros of ExtendableSparseMatrix.
"""
function SparseArrays.nnz(ext::ExtendableSparseMatrix)
    ennz=0
    if ext.lnkmatrix!=nothing
        ennz=nnz(ext.lnkmatrix)
    end
    return nnz(ext.cscmatrix)+ennz
end



# Struct holding pair of value and row
# number, for sorting
struct ColEntry{Tv,Ti<:Integer}
    rowval::Ti
    nzval::Tv
end

# Comparison method for sorting
Base.isless(x::ColEntry{Tv, Ti},y::ColEntry{Tv, Ti}) where {Tv,Ti<:Integer} = (x.rowval<y.rowval)

"""
$(SIGNATURES)

Create new CSC matrix with sorted entries from a SparseMatrixCSC  csc and 
[`SparseMatrixLNK`](@ref)  lnk.

This method assumes that there are no entries with the same
indices in lnk and csc, therefore  it appears too dangerous for general use and
so we don't export it.  A generalization appears to be possible, though.
"""
function _splice(lnk::SparseMatrixLNK{Tv,Ti},csc::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti<:Integer}
    @assert(csc.m==lnk.m)
    @assert(csc.n==lnk.n)

    xnnz=nnz(csc)+nnz(lnk)
    colptr=Vector{Ti}(undef,csc.n+1)
    rowval=Vector{Ti}(undef,xnnz)
    nzval=Vector{Tv}(undef,xnnz)

    # Detect the maximum column length of lnk
    lnk_maxcol=0
    for j=1:csc.n
        lcol=0
        k=j
        while k>0
            lcol+=1
            k=lnk.colptr[k]
        end
        lnk_maxcol=max(lcol,lnk_maxcol)
    end
    
    # pre-allocate column  data
    col=[ColEntry{Tv,Ti}(0,0) for i=1:lnk_maxcol]


    
    inz=1 # counts the nonzero entries in the new matrix

    # loop over all columns
    for j=1:csc.n
        # put extension entries into col and sort them
        k=j
        l_lnk_col=0
        while k>0
            if lnk.rowval[k]>0
                l_lnk_col+=1
                col[l_lnk_col]=ColEntry(lnk.rowval[k],lnk.nzval[k])
            end
            k=lnk.colptr[k]
        end
        sort!(col,1,l_lnk_col, Base.QuickSort, Base.Forward)


        # jointly sort lnk and csc entries  into new matrix data
        colptr[j]=inz
        jcol=1 # counts the entries in col
        jcsc=csc.colptr[j]  # counts entries in csc
        while true
            
            # Insert entries from csc into new structure
            # if the row number is before col[jcol]
            if ( nnz(csc)>0 &&  (jcsc<csc.colptr[j+1]) ) && 
                 (
                     (jcol<=l_lnk_col && csc.rowval[jcsc]<col[jcol].rowval) || 
                     (jcol>l_lnk_col)
                 )
                rowval[inz]=csc.rowval[jcsc]
                nzval[inz]=csc.nzval[jcsc]
                jcsc+=1
                inz+=1

            # Otherwise, insert next entry from col    
            elseif jcol<=l_lnk_col
                rowval[inz]=col[jcol].rowval
                nzval[inz]=col[jcol].nzval
                jcol+=1
                inz+=1
            else
                break
            end
        end
    end
    colptr[csc.n+1]=inz
    SparseMatrixCSC{Tv,Ti}(csc.m,csc.n,colptr,rowval,nzval)
end


"""
$(SIGNATURES)

If there are new entries in extension, create new CSC matrix using [`_splice`](@ref)
and reset linked list based extension.
"""
function flush!(ext::ExtendableSparseMatrix{Tv,Ti}) where {Tv, Ti<:Integer}
    if ext.lnkmatrix!=nothing && nnz(ext.lnkmatrix)>0
        ext.cscmatrix=_splice(ext.lnkmatrix,ext.cscmatrix)
        ext.lnkmatrix=nothing
    end
    return ext
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
