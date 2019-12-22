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
function ExtendableSparseMatrix(::Type{Tv},::Type{Ti},m::Integer, n::Integer) where{Tv,Ti<:Integer}
    ExtendableSparseMatrix{Tv,Ti}(m,n)
end

"""
$(SIGNATURES)

Create empty ExtendablSparseMatrix.
This is a pendant to spzeros.
"""
ExtendableSparseMatrix(::Type{Tv},m::Integer, n::Integer) where{Tv}=ExtendableSparseMatrix{Tv,Int}(m,n)


"""
$(SIGNATURES)

Create empty ExtendableSparseMatrix.
This is a pendant to spzeros.
"""
ExtendableSparseMatrix(m::Integer, n::Integer)=ExtendableSparseMatrix{Float64,Int}(m,n)


"""
$(SIGNATURES)

  Create ExtendableSparseMatrix from sparse matrix
"""
function ExtendableSparseMatrix(M::SparseMatrixCSC{Tv,Ti}) where{Tv,Ti<:Integer}
    return ExtendableSparseMatrix{Tv,Ti}(M, nothing)
end



"""
$(SIGNATURES)

Return index corresponding to entry [i,j] in the array of nonzeros,
if the entry exists, otherwise, return 0.
"""
function findindex(S::SparseMatrixCSC{T}, i::Integer, j::Integer) where T
    if !(1 <= i <= S.m && 1 <= j <= S.n); throw(BoundsError()); end
    r1 = Int(S.colptr[j])
    r2 = Int(S.colptr[j+1]-1)
    if r1>r2
        return 0
    end

    # See sparsematrix.jl
    r1 = searchsortedfirst(S.rowval, i, r1, r2, Base.Forward)
    if (r1>r2 ||S.rowval[r1] != i)
        return 0
    end
    return r1
end



"""
$(SIGNATURES)

Find index in CSC matrix and set value if it exists. Otherwise,
set index in extension.
"""
function Base.setindex!(M::ExtendableSparseMatrix{Tv,Ti}, v, i::Integer, j::Integer) where{Tv,Ti<:Integer}
    k=findindex(M.cscmatrix,i,j)
    if k>0
        M.cscmatrix.nzval[k]=v
    else
        if M.lnkmatrix==nothing
            M.lnkmatrix=SparseMatrixLNK{Tv, Ti}(M.cscmatrix.m, M.cscmatrix.n)
        end
        M.lnkmatrix[i,j]=v
    end
end





"""
$(SIGNATURES)

Find index in CSC matrix and return value, if it exists.
Otherwise, return value from extension.
"""
function Base.getindex(M::ExtendableSparseMatrix{Tv,Ti},i::Integer, j::Integer) where{Tv,Ti<:Integer}
    k=findindex(M.cscmatrix,i,j)
    if k>0
        return M.cscmatrix.nzval[k]
    elseif M.lnkmatrix==nothing
        return zero(Tv)
    else
        return M.lnkmatrix[i,j]
    end
end

"""
$(SIGNATURES)

Size of ExtendableSparseMatrix.
"""
Base.size(E::ExtendableSparseMatrix) = (E.cscmatrix.m, E.cscmatrix.n)


"""
$(SIGNATURES)

Number of nonzeros of ExtendableSparseMatrix.
"""
function SparseArrays.nnz(E::ExtendableSparseMatrix)
    ennz=0
    if E.lnkmatrix!=nothing
        ennz=nnz(E.lnkmatrix)
    end
    return nnz(E.cscmatrix)+ennz
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
and reset extension.
"""
function flush!(M::ExtendableSparseMatrix{Tv,Ti}) where {Tv, Ti<:Integer}
    if M.lnkmatrix!=nothing && nnz(M.lnkmatrix)>0
        M.cscmatrix=_splice(M.lnkmatrix,M.cscmatrix)
        M.lnkmatrix=nothing
    end
    return M
end



"""
$(SIGNATURES)

Flush and delegate to cscmatrix.
"""
function SparseArrays.nonzeros(E::ExtendableSparseMatrix)
    @inbounds flush!(E)
    return nonzeros(E.cscmatrix)
end



"""
$(SIGNATURES)

Flush and delegate to cscmatrix.
"""
function SparseArrays.rowvals(E::ExtendableSparseMatrix)
    @inbounds flush!(E)
    return rowvals(E.cscmatrix)
end


"""
$(SIGNATURES)

Flush and delegate to cscmatrix.
"""
function colptrs(E::ExtendableSparseMatrix)
    @inbounds flush!(E)
    return E.cscmatrix.colptr
end


"""
$(SIGNATURES)

Flush and delegate to cscmatrix.
"""
function SparseArrays.findnz(E::ExtendableSparseMatrix)
    @inbounds flush!(E)
    return findnz(E.cscmatrix)
end


"""
$(SIGNATURES)

Delegating LU factorization.
"""
function LinearAlgebra.lu(E::ExtendableSparseMatrix)
    @inbounds flush!(E)
    return LinearAlgebra.lu(E.cscmatrix)
end

"""
$(SIGNATURES)

Delegating Matrix multiplication
"""
function  LinearAlgebra.mul!(r::AbstractArray{T,1} where T, E::ExtendableSparse.ExtendableSparseMatrix, x::AbstractArray{T,1} where T)
    @inbounds flush!(E)
    return LinearAlgebra.mul!(r,E.cscmatrix,x)
end


"""
$(SIGNATURES)

Delegating Matrix ldiv
"""
function  LinearAlgebra.ldiv!(r::AbstractArray{T,1} where T, E::ExtendableSparse.ExtendableSparseMatrix, x::AbstractArray{T,1} where T)
    @inbounds flush!(E)
    return LinearAlgebra.ldiv!(r,E.cscmatrix,x)
end

"""
$(SIGNATURES)

Delegating Matrix multiplication
"""
function  LinearAlgebra.mul!(r::AbstractArray{T,2} where T, E::ExtendableSparse.ExtendableSparseMatrix, x::AbstractArray{T,2} where T)
    @inbounds flush!(E)
    return LinearAlgebra.mul!(r,E.cscmatrix,x)
end


"""
$(SIGNATURES)

Delegating Matrix ldiv
"""
function  LinearAlgebra.ldiv!(r::AbstractArray{T,2} where T, E::ExtendableSparse.ExtendableSparseMatrix, x::AbstractArray{T,2} where T)
    @inbounds flush!(E)
    return LinearAlgebra.ldiv!(r,E.cscmatrix,x)
end
