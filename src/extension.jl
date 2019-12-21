
"""
$(TYPEDEF)

Struct to hold extension data of sparse matrix.

$(TYPEDFIELDS)
"""
mutable struct SparseMatrixExtension{Tv,Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti}

    """
    Number of rows
    """
    m::Ti


    """
    Number of columns
    """
    n::Ti


    """
    Number of nonzeros
    """
    nnz::Ti


    """
    Linked list of column entries. Initial length is n,
    it grows with each new entry.
    
    colptr[index] contains the next
    index in the list or zero, terminating the list which
    starts at index 1<=j<=n for each column j.
    """
    colptr::Vector{Ti}      


    """
    Row numbers. For each index it contains the zero (initial state)
    or the row numbers corresponding to the column entry list in colptr.

    Initial length is n,
    it grows with each new entry.
    """
    rowval::Vector{Ti}

    """
    Nonzero entry values correspondin to each pair
    (colptr[index],rowval[index])

    Initial length is n,  it grows with each new entry.
    """
    nzval::Vector{Tv}       
end



"""
$(SIGNATURES)
    
Constructor of empty extension
"""
SparseMatrixExtension{Tv,Ti}(m::Integer, n::Integer)  where {Tv,Ti<:Integer} =    SparseMatrixExtension{Tv,Ti}(m,n,0,zeros(Ti,n),zeros(Ti,n),zeros(Tv,n))




"""
$(SIGNATURES)
    
Return value stored for entry or zero if not found
"""
function Base.getindex(E::SparseMatrixExtension{Tv,Ti},i::Integer, j::Integer) where {Tv,Ti<:Integer}

    if !((1 <= i <= E.m) & (1 <= j <= E.n))
        throw(BoundsError(E, (i,j)))
    end
    
    k=j
    while k>0
        if E.rowval[k]==i
            return E.nzval[k]
        end
        k=E.colptr[k]
    end

    return zero(Tv)
end



"""
$(SIGNATURES)
    
Update value of existing entry, otherwise extend matrix.
"""
function Base.setindex!(E::SparseMatrixExtension{Tv,Ti}, _v, _i::Integer, _j::Integer) where {Tv,Ti<:Integer}
    v = convert(Tv, _v)
    i = convert(Ti, _i)
    j = convert(Ti, _j)

    if !((1 <= i <= E.m) & (1 <= j <= E.n))
        throw(BoundsError(E, (i,j)))
    end
    
    # Set the first  column entry if it was not yet set.
    if E.rowval[j]==0
        E.rowval[j]=i       
        E.nzval[j]=v
        E.nnz+=1
        return E
    end

    # Traverse list for existing entry
    k=j
    k0=j
    while k>0
        # Update value and return if entry has been found
        if E.rowval[k]==i
            E.nzval[k]=v
            return E
        end
        k0=k
        # Next element in the list
        k=E.colptr[k]
    end

    # Append entry if not found
    push!(E.nzval,v)
    push!(E.rowval,i)

    # Shift the end of the list
    push!(E.colptr,0)
    E.colptr[k0]=length(E.nzval)

    # Update number of nonzero entries
    E.nnz+=1
    return E
end


"""
$(SIGNATURES)

Return tuple containing size of the matrix.
"""
Base.size(E::SparseMatrixExtension) = (E.m, E.n)


"""
$(SIGNATURES)

Return number of nonzero entries.
"""
SparseArrays.nnz(E::SparseMatrixExtension)=E.nnz


"""
$(SIGNATURES)

Dummy flush! method for Sparse matrix extension. Just
used in thest methods
"""
function flush!(M::SparseMatrixExtension{Tv, Ti}) where{Tv, Ti}
    return M
end

