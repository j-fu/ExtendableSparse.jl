
"""
$(TYPEDEF)

Struct to hold sparse matrix in the linked list format.

Modeled after the linked list sparse matrix format described in 
the  [whitepaper](https://www-users.cs.umn.edu/~saad/software/SPARSKIT/paper.ps)
and the  [SPARSEKIT2 source code](https://www-users.cs.umn.edu/~saad/software/SPARSKIT/SPARSKIT2.tar.gz)
by Y. Saad. He writes "This is one of the oldest data structures used for sparse matrix computations."

The relevant source [formats.f](https://salsa.debian.org/science-team/sparskit/blob/master/FORMATS/formats.f)
is also available in the debian/science gitlab.

Probably this format was around already in SPARSPAK by E.Chu, A.George and J.Liu, however this is 
hard to verify, as it indeed appears that the source code of SPARSPAK [vanished from the internet](http://www.netlib.org/sparspak/readme).

The advantage of the linked list structure is the fact that upon insertion
of a new entry, the arrays describing the structure grow at their respective ends and
can be conveniently updated via `push!`.  No copying of existing data is necessary.

$(TYPEDFIELDS)
"""
mutable struct SparseMatrixLNK{Tv,Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti}

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
    index in the list or zero, in the later case terminating the list which
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
SparseMatrixLNK{Tv,Ti}(m::Integer, n::Integer)  where {Tv,Ti<:Integer} =    SparseMatrixLNK{Tv,Ti}(m,n,0,zeros(Ti,n),zeros(Ti,n),zeros(Tv,n))




"""
$(SIGNATURES)
    
Return value stored for entry or zero if not found
"""
function Base.getindex(lnk::SparseMatrixLNK{Tv,Ti},i::Integer, j::Integer) where {Tv,Ti<:Integer}

    if !((1 <= i <= lnk.m) & (1 <= j <= lnk.n))
        throw(BoundsError(lnk, (i,j)))
    end
    
    k=j
    while k>0
        if lnk.rowval[k]==i
            return lnk.nzval[k]
        end
        k=lnk.colptr[k]
    end

    return zero(Tv)
end



"""
$(SIGNATURES)
    
Update value of existing entry, otherwise extend matrix.
"""
function Base.setindex!(lnk::SparseMatrixLNK{Tv,Ti}, _v, _i::Integer, _j::Integer) where {Tv,Ti<:Integer}
    v = convert(Tv, _v)
    i = convert(Ti, _i)
    j = convert(Ti, _j)

    if !((1 <= i <= lnk.m) & (1 <= j <= lnk.n))
        throw(BoundsError(lnk, (i,j)))
    end
    
    # Set the first  column entry if it was not yet set.
    if lnk.rowval[j]==0
        lnk.rowval[j]=i       
        lnk.nzval[j]=v
        lnk.nnz+=1
        return lnk
    end

    # Traverse list for existing entry
    k=j
    k0=j
    while k>0
        # Update value and return if entry has been found
        if lnk.rowval[k]==i
            lnk.nzval[k]=v
            return lnk
        end
        k0=k
        # Next element in the list
        k=lnk.colptr[k]
    end

    # Append entry if not found
    push!(lnk.nzval,v)
    push!(lnk.rowval,i)

    # Shift the end of the list
    push!(lnk.colptr,0)
    lnk.colptr[k0]=length(lnk.nzval)

    # Update number of nonzero entries
    lnk.nnz+=1
    return lnk
end


"""
$(SIGNATURES)

Return tuple containing size of the matrix.
"""
Base.size(lnk::SparseMatrixLNK) = (lnk.m, lnk.n)


"""
$(SIGNATURES)

Return number of nonzero entries.
"""
SparseArrays.nnz(lnk::SparseMatrixLNK)=lnk.nnz


"""
$(SIGNATURES)

Dummy flush! method for Sparse matrix extension. Just
used in test methods
"""
function flush!(lnk::SparseMatrixLNK{Tv, Ti}) where{Tv, Ti}
    return lnk
end

