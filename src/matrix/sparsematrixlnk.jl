
"""
    $(TYPEDEF)
Trivial dictionary for fixed size, single threaded case.
"""
struct TrivialDict{Ti} end
Base.getindex(::TrivialDict,i)=i
Base.haskey(::TrivialDict,k)=true
Base.setindex!(::TrivialDict,v,i)=nothing
Base.copy(::TrivialDict{Ti}) where Ti=TrivialDict{Ti}()

"""
$(TYPEDEF)

Struct to hold sparse matrix in the linked list format.

Modeled after the linked list sparse matrix format described in 
the  [whitepaper](https://www-users.cs.umn.edu/~saad/software/SPARSKIT/paper.ps)
and the  [SPARSEKIT2 source code](https://www-users.cs.umn.edu/~saad/software/SPARSKIT/SPARSKIT2.tar.gz)
by Y. Saad. He writes "This is one of the oldest data structures used for sparse matrix computations."

The relevant source [formats.f](https://salsa.debian.org/science-team/sparskit/blob/master/FORMATS/formats.f)
is also available in the debian/science gitlab.

The advantage of the linked list structure is the fact that upon insertion
of a new entry, the arrays describing the structure can grow at their respective ends and
can be conveniently updated via `push!`.  No copying of existing data is necessary.


$(TYPEDFIELDS)
"""
mutable struct SparseMatrixLNK{Tv, Ti <: Integer} <: AbstractSparseMatrix{Tv, Ti}
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
    Number of entry storage locations
    """
    nstored::Ti

    """
    coldict[n] points to the index in colptr where column n starts.
    """
    coldict::Union{Dict{Ti,Ti},TrivialDict{Ti}}
    
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


global _flexsize::Bool=false

flexsize!(b::Bool)=  global _flexsize=b

"""
$(SIGNATURES)
    
Constructor of empty matrix.
"""
function SparseMatrixLNK{Tv, Ti}(m, n) where {Tv, Ti <: Integer}
    global _flexsize
    if _flexsize
        SparseMatrixLNK{Tv, Ti}(m, n, 0, 0, Dict{Ti,Ti}(), zeros(Ti, 0), zeros(Ti, 0), zeros(Tv, 0))
    else
        SparseMatrixLNK{Tv, Ti}(m, n, 0, n, TrivialDict{Ti}(), zeros(Ti, n), zeros(Ti,n), zeros(Tv, n))
    end
end

"""
$(SIGNATURES)
    
Constructor of empty matrix.
"""
function SparseMatrixLNK(valuetype::Type{Tv}, indextype::Type{Ti}, m,
                         n) where {Tv, Ti <: Integer}
    SparseMatrixLNK{Tv, Ti}(m, n)
end

"""
$(SIGNATURES)
    
Constructor of empty matrix.
"""
SparseMatrixLNK(valuetype::Type{Tv}, m, n) where {Tv} = SparseMatrixLNK(Tv, Int, m, n)

"""
$(SIGNATURES)
    
Constructor of empty matrix.
"""
SparseMatrixLNK(m, n) = SparseMatrixLNK(Float64, m, n)

"""
$(SIGNATURES)
    
Constructor from SparseMatrixCSC.

"""
function SparseMatrixLNK(csc::SparseArrays.SparseMatrixCSC{Tv, Ti}) where {Tv, Ti <:
                                                                               Integer}
    lnk = SparseMatrixLNK{Tv, Ti}(csc.m, csc.n)
    for j = 1:(csc.n)
        for k = csc.colptr[j]:(csc.colptr[j + 1] - 1)
            lnk[csc.rowval[k], j] = csc.nzval[k]
        end
    end
    lnk
end

function findindex(lnk::SparseMatrixLNK, i, j)
    if !((1 <= i <= lnk.m) & (1 <= j <= lnk.n))
        throw(BoundsError(lnk, (i, j)))
    end

    if haskey(lnk.coldict,j)
        k = lnk.coldict[j]
        k0 = k
        while k > 0
            if lnk.rowval[k] == i
                return k, 0
            end
            k0 = k
            k = lnk.colptr[k]
        end
        return 0, k0
    else
        return 0,0
    end
end

"""
$(SIGNATURES)
    
Return value stored for entry or zero if not found
"""
function Base.getindex(lnk::SparseMatrixLNK{Tv, Ti}, i, j) where {Tv, Ti}
    k, k0 = findindex(lnk, i, j)
    if k == 0
        return zero(Tv)
    else
        return lnk.nzval[k]
    end
end

function increase_storage!(lnk::SparseMatrixLNK)
    lnk.nstored+=1
    if length(lnk.nzval) < lnk.nstored
        newsize = Int(ceil(5.0 * lnk.nstored / 4.0))
        resize!(lnk.nzval, newsize)
        resize!(lnk.rowval, newsize)
        resize!(lnk.colptr, newsize)
    end
end


function addentry!(lnk::SparseMatrixLNK, coldict::TrivialDict, i, j, k0)
    # increase number of entries
    lnk.nnz += 1
    # Append entry if not found
    if lnk.rowval[j]==0
        lnk.rowval[j] = i
        return j
    else
        increase_storage!(lnk)

        # Shift the end of the list
        lnk.rowval[lnk.nstored]=i
        lnk.colptr[lnk.nstored] = 0
        lnk.colptr[k0] = lnk.nstored
        return lnk.nstored
    end
end

function addentry!(lnk::SparseMatrixLNK, coldict::Dict, i, j, k0)
    # increase number of entries

    increase_storage!(lnk)

    lnk.nnz+=1

    # Append entry if not found
    lnk.rowval[lnk.nstored] = i

    if k0==0
        coldict[j]=lnk.nstored
        lnk.colptr[lnk.nstored] = 0
    else
        # Shift the end of the list
        lnk.colptr[lnk.nstored] = 0
        lnk.colptr[k0] = lnk.nstored
    end
    return lnk.nstored
end

"""
$(SIGNATURES)
    
Update value of existing entry, otherwise extend matrix if v is nonzero.
"""
function Base.setindex!(lnk::SparseMatrixLNK, v, i, j)
    if !((1 <= i <= lnk.m) & (1 <= j <= lnk.n))
        throw(BoundsError(lnk, (i, j)))
    end
    k, k0 = findindex(lnk, i, j)
    if k > 0
        lnk.nzval[k] = v
        return lnk
    end
    if !iszero(v)
        k = addentry!(lnk,lnk.coldict, i, j, k0)
        lnk.nzval[k] = v
    end
    return lnk
end

"""
$(SIGNATURES)

Update element of the matrix  with operation `op`. 
It assumes that `op(0,0)==0`. If `v` is zero, no new 
entry is created.
"""
function updateindex!(lnk::SparseMatrixLNK{Tv, Ti}, op, v, i, j) where {Tv, Ti}
    k, k0 = findindex(lnk, i, j)
    if k > 0
        lnk.nzval[k] = op(lnk.nzval[k], v)
        return lnk
    end
    if !iszero(v)
        k = addentry!(lnk,lnk.coldict, i, j, k0)
        lnk.nzval[k] = op(zero(Tv), v)
    end
    lnk
end

"""
$(SIGNATURES)

Update element of the matrix  with operation `op`. 
It assumes that `op(0,0)==0`. If `v` is zero a new entry
is created nevertheless.
"""
function rawupdateindex!(lnk::SparseMatrixLNK{Tv, Ti}, op, v, i, j) where {Tv, Ti}
    k, k0 = findindex(lnk, i, j)
    if k > 0
        lnk.nzval[k] = op(lnk.nzval[k], v)
    else
        k = addentry!(lnk,lnk.coldict, i, j, k0)
        lnk.nzval[k] = op(zero(Tv), v)
    end
    lnk
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
SparseArrays.nnz(lnk::SparseMatrixLNK) = lnk.nnz

"""
$(SIGNATURES)

Dummy flush! method for SparseMatrixLNK. Just
used in test methods
"""
function flush!(lnk::SparseMatrixLNK{Tv, Ti}) where {Tv, Ti}
    return lnk
end

# Struct holding pair of value and row
# number, for sorting
struct ColEntry{Tv, Ti <: Integer}
    rowval::Ti
    nzval::Tv
end

# Comparison method for sorting
Base.isless(x::ColEntry, y::ColEntry) = (x.rowval < y.rowval)

"""
$(SIGNATURES)

Add SparseMatrixCSC matrix and [`SparseMatrixLNK`](@ref)  lnk, returning a SparseMatrixCSC
"""
function Base.:+(lnk::SparseMatrixLNK{Tv, Ti},
                 csc::SparseMatrixCSC)::SparseMatrixCSC where {Tv, Ti <: Integer}
    @assert(csc.m==lnk.m)
    @assert(csc.n==lnk.n)
    if nnz(lnk)==0
        return copy(csc)
    end
    
    # overallocate arrays in order to avoid
    # presumably slower push!
    xnnz = nnz(csc) + nnz(lnk)
    colptr = Vector{Ti}(undef, csc.n + 1)
    rowval = Vector{Ti}(undef, xnnz)
    nzval = Vector{Tv}(undef, xnnz)

    # Detect the maximum column length of lnk
    lnk_maxcol = 0
    for j = 1:(csc.n)
        lcol = zero(Ti)
        if haskey(lnk.coldict,j)
            k = lnk.coldict[j]
            while k > 0
                lcol += 1
                k = lnk.colptr[k]
            end
            lnk_maxcol = max(lcol, lnk_maxcol)
        end
    end

    # pre-allocate column  data
    col = [ColEntry{Tv, Ti}(0, zero(Tv)) for i = 1:lnk_maxcol]

    inz = 1 # counts the nonzero entries in the new matrix

    in_csc_col(jcsc, j) = (nnz(csc) > zero(Ti)) && (jcsc < csc.colptr[j + 1])

    in_lnk_col(jlnk, l_lnk_col) = (jlnk <= l_lnk_col)

    # loop over all columns
    for j = 1:(csc.n)
        # Copy extension entries into col and sort them
        l_lnk_col = 0
        if haskey(lnk.coldict,j)
            k = lnk.coldict[j]
            while k > 0
                if lnk.rowval[k] > 0
                    l_lnk_col += 1
                    col[l_lnk_col] = ColEntry(lnk.rowval[k], lnk.nzval[k])
                end
                k = lnk.colptr[k]
            end
            sort!(col, 1, l_lnk_col, Base.QuickSort, Base.Forward)
        end
        
        # jointly sort lnk and csc entries  into new matrix data
        # this could be replaced in a more transparent manner by joint sorting:
        # make a joint array for csc and lnk col, sort them.
        # Will this be faster? 
        colptr[j] = inz
        jlnk = one(Ti) # counts the entries in col
        jcsc = csc.colptr[j]  # counts entries in csc
        
        while true
            if in_csc_col(jcsc, j) &&
                (in_lnk_col(jlnk, l_lnk_col) && csc.rowval[jcsc] < col[jlnk].rowval ||
                 !in_lnk_col(jlnk, l_lnk_col))
                # Insert entries from csc into new structure
                rowval[inz] = csc.rowval[jcsc]
                nzval[inz] = csc.nzval[jcsc]
                jcsc += 1
                inz += 1
            elseif in_csc_col(jcsc, j) &&
                (in_lnk_col(jlnk, l_lnk_col) && csc.rowval[jcsc] == col[jlnk].rowval)
                # Add up entries from csc and lnk
                rowval[inz] = csc.rowval[jcsc]
                nzval[inz] = csc.nzval[jcsc] + col[jlnk].nzval
                jcsc += 1
                inz += 1
                jlnk += 1
            elseif in_lnk_col(jlnk, l_lnk_col)
                # Insert entries from lnk res. col into new structure
                rowval[inz] = col[jlnk].rowval
                nzval[inz] = col[jlnk].nzval
                jlnk += 1
                inz += 1
            else
                break
            end
        end
    end
    colptr[csc.n + 1] = inz
    # Julia 1.7 wants this correct
    resize!(rowval, inz - 1)
    resize!(nzval, inz - 1)
    SparseMatrixCSC{Tv, Ti}(csc.m, csc.n, colptr, rowval, nzval)
end

Base.:+(csc::SparseMatrixCSC, lnk::SparseMatrixLNK) = lnk + csc

"""
$(SIGNATURES)
    
Constructor from SparseMatrixLNK.

"""
function SparseArrays.SparseMatrixCSC(lnk::SparseMatrixLNK)::SparseMatrixCSC
    csc = spzeros(lnk.m, lnk.n)
    lnk + csc
end

rowvals(S::SparseMatrixLNK) = getfield(S, :rowval)
getcolptr(S::SparseMatrixLNK) = getfield(S, :colptr)
nonzeros(S::SparseMatrixLNK) = getfield(S, :nzval)

function Base.copy(S::SparseMatrixLNK)
    SparseMatrixLNK(size(S, 1),
                    size(S, 2),
                    S.nnz,
                    S.nstored,
                    copy(S.coldict),
                    copy(getcolptr(S)),
                    copy(rowvals(S)),
                    copy(nonzeros(S)))
end
