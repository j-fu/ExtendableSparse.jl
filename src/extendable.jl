##################################################################
"""
$(TYPEDEF)

Extendable sparse matrix. A nonzero  entry of this matrix is contained
either in cscmatrix, or in extmatrix, never in both.

$(TYPEDFIELDS)
"""
mutable struct ExtendableSparseMatrix{Tv,Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti}
    """
    Final matrix data
    """
    cscmatrix::SparseMatrixCSC{Tv,Ti}

    """
    Intermediate structure holding data of extension
    """
    extmatrix::SparseMatrixExtension{Tv,Ti}
end


"""
$(TYPEDSIGNATURES)

Create empty ExtendablSparseMatrix.

This is a pendant to spzeros.
"""
ExtendableSparseMatrix{Tv,Ti}(m::Integer, n::Integer) where {Tv,Ti<:Integer}=ExtendableSparseMatrix{Tv,Ti}(spzeros(Tv,Ti,m,n),SparseMatrixExtension{Tv, Ti}(m,n))


"""
$(TYPEDSIGNATURES)

Return index corresponding to entry (i,j) in the array of nonzeros,
if the entry exists, otherwise, return 0.
"""
function findindex(S::SparseMatrixCSC{T}, i::Integer, j::Integer) where T
    if !(1 <= i <= S.m && 1 <= j <= S.n); throw(BoundsError()); end
    r1 = Int(S.colptr[j])
    r2 = Int(S.colptr[j+1]-1)
    if r1>r2
        return zero(T)
    end

    # See sparsematrix.jl
    r1 = searchsortedfirst(S.rowval, i, r1, r2, Base.Forward)
    if (r1>r2 ||S.rowval[r1] != i)
        return zero(T)
    end
    return r1
end



"""
$(SIGNATURES)

Find index in CSC matrix and set value if it exists. Otherwise,
set index in extension.
"""
function Base.setindex!(M::ExtendableSparseMatrix, v, i::Integer, j::Integer)
    k=findindex(M.cscmatrix,i,j)
    if k>0
        M.cscmatrix.nzval[k]=v
    else
        M.extmatrix[i,j]=v
    end
end





"""
$(SIGNATURES)

Find index in CSC matrix and return value, if it exists.
Otherwise, return value from extension.
"""
function Base.getindex(M::ExtendableSparseMatrix,i::Integer, j::Integer)
    k=findindex(M.cscmatrix,i,j)
    if k>0
        return M.cscmatrix.nzval[k]
    else
        return M.extmatrix[i,j]
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
SparseArrays.nnz(E::ExtendableSparseMatrix)=(nnz(E.cscmatrix)+nnz(E.extmatrix))



# Struct holding pair of value and row
# number, for sorting
struct ColEntry{Tv,Ti<:Integer}
    i::Ti
    v::Tv
end

# Comparison method for sorting
Base.isless(x::ColEntry{Tv, Ti},y::ColEntry{Tv, Ti}) where {Tv,Ti<:Integer} = (x.i<y.i)


function _splice(E::SparseMatrixExtension{Tv,Ti},S::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti<:Integer}
    # Create new CSC matrix with sorted entries from  CSC matrix S and matrix extension E.
    #
    # This method assumes that there are no entries with the same
    # indices in E and S, therefore  it appears too dangerous for general use and
    # so we don't export it. Generalizations appear to be possible, though.

    @assert(S.m==E.m)
    @assert(S.n==E.n)

    xnnz=nnz(S)+nnz(E)
    colptr=Vector{Ti}(undef,S.n+1)
    rowval=Vector{Ti}(undef,xnnz)
    nzval=Vector{Tv}(undef,xnnz)

    # Detect the maximum column length of E
    E_maxcol=0
    for j=1:S.n
        lcol=0
        k=j
        while k>0
            lcol+=1
            k=E.colptr[k]
        end
        E_maxcol=max(lcol,E_maxcol)
    end

    # pre-allocate column 
    col=[ColEntry{Tv,Ti}(0,0) for i=1:E_maxcol]


    
    inz=1 # counts the nonzero entries in the new matrix

    # loop over all columns
    for j=1:S.n
        # put extension entries into col and sort them
        k=j
        lxcol=0
        while k>0
            if E.rowval[k]>0
                lxcol+=1
                col[lxcol]=ColEntry(E.rowval[k],E.nzval[k])
            end
            k=E.colptr[k]
        end
        sort!(col,1,lxcol, Base.QuickSort, Base.Forward)


        # jointly sort E and S entries  into new matrix data
        colptr[j]=inz
        jcol=1 # counts the entries in col
        k=S.colptr[j]  # counts entries in S
        while true

            # Check if there are entries in S preceding col[jcol]
            if ( nnz(S)>0 &&  (k<S.colptr[j+1]) ) && 
                 (
                     (jcol<=lxcol && S.rowval[k]<col[jcol].i) || 
                     (jcol>lxcol)
                 )
                rowval[inz]=S.rowval[k]
                nzval[inz]=S.nzval[k]
                k+=1
                inz+=1

            # Otherwise, insert next entry from col    
            elseif jcol<=lxcol
                rowval[inz]=col[jcol].i
                nzval[inz]=col[jcol].v
                jcol+=1
                inz+=1
            else
                break
            end
        end
    end
    colptr[S.n+1]=inz
    S1=SparseMatrixCSC{Tv,Ti}(S.m,S.n,colptr,rowval,nzval)
    return S1
end


"""
$(TYPEDSIGNATURES)

If there are new entries in extension, create new CSC matrix
and reset extension.
"""
function flush!(M::ExtendableSparseMatrix{Tv,Ti}) where {Tv, Ti<:Integer}
    if nnz(M.extmatrix)>0
        M.cscmatrix=_splice(M.extmatrix,M.cscmatrix)
        M.extmatrix=SparseMatrixExtension{Tv,Ti}(M.cscmatrix.m, M.cscmatrix.n)
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
function xcolptrs(E::ExtendableSparseMatrix)
    @inbounds flush!(E)
    return E.cscmatrix.colptr
end


"""
$(SIGNATURES)

Flush and delegate to cscmatrix.
"""
function colptrs(E::ExtendableSparseMatrix)
    flush!(E)
    return E.cscmatrix.colptr
end


"""
$(SIGNATURES)

Flush and delegate to cscmatrix.
"""
function SparseArrays.findnz(E::ExtendableSparseMatrix)
    flush!(E)
    return findnz(E.cscmatrix)
end


"""
$(TYPEDSIGNATURES)

Drop in replacement for LU factorization.

"""
function LinearAlgebra.lu(E::ExtendableSparseMatrix)
    @inbounds flush!(E)
    return LinearAlgebra.lu(E.cscmatrix)
end




"""
$(SIGNATURES)

Flush and delegate to cscmatrix.
"""
function LinearAlgebra.mul!(r::AbstractArray{Tv,1},
                            E::ExtendableSparse.ExtendableSparseMatrix{Tv,Ti},
                            x::AbstractArray{Tv,1}) where{Tv,Ti<:Integer}
    @inbounds flush!(E)
    return LinearAlgebra.mul!(r,E.cscmatrix,x)
end


