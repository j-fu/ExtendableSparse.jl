"""
    $(TYPEDEF)

Modification of SparseMatrixLNK where the pointer to first index of
column j is stored in a dictionary.
"""
mutable struct SparseMatrixLNKDict{Tv, Ti <: Integer} <: AbstractSparseMatrix{Tv, Ti}
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
    Length of arrays
    """
    nentries::Ti

    """
    Linked list of column entries. Initial length is n,
    it grows with each new entry.

    colptr[index] contains the next
    index in the list or zero, in the later case terminating the list which
    starts at index 1<=j<=n for each column j.
    """
    colptr::Vector{Ti}

    """
    Dictionary to store start indices of columns
    """
    colstart::Dict{Ti,Ti}

    """
    Row numbers. For each index it contains the zero (initial state)
    or the row numbers corresponding to the column entry list in colptr.
    """
    rowval::Vector{Ti}

    """
    Nonzero entry values correspondin to each pair
    (colptr[index],rowval[index])
    """
    nzval::Vector{Tv}
end

"""
$(SIGNATURES)
    
Constructor of empty matrix.
"""
function SparseMatrixLNKDict{Tv, Ti}(m, n) where {Tv, Ti <: Integer}
    SparseMatrixLNKDict{Tv, Ti}(m, n, 0, 0,  zeros(Ti,10), Dict{Ti,Ti}(), zeros(Ti,10), zeros(Ti,10))
end

"""
$(SIGNATURES)
    
Constructor of empty matrix.
"""
function SparseMatrixLNKDict(valuetype::Type{Tv}, indextype::Type{Ti}, m,
                         n) where {Tv, Ti <: Integer}
    SparseMatrixLNKDict{Tv, Ti}(m, n)
end

"""
$(SIGNATURES)
    
Constructor of empty matrix.
"""
SparseMatrixLNKDict(valuetype::Type{Tv}, m, n) where {Tv} = SparseMatrixLNKDict(Tv, Int, m, n)

"""
$(SIGNATURES)
    
Constructor of empty matrix.
"""
SparseMatrixLNKDict(m, n) = SparseMatrixLNKDict(Float64, m, n)

"""
$(SIGNATURES)
    
Constructor from SparseMatrixCSC.

"""
function SparseMatrixLNKDict(csc::SparseArrays.SparseMatrixCSC{Tv, Ti}) where {Tv, Ti <:
                                                                               Integer}
    lnk = SparseMatrixLNKDict{Tv, Ti}(csc.m, csc.n)
    for j = 1:(csc.n)
        for k = csc.colptr[j]:(csc.colptr[j + 1] - 1)
            lnk[csc.rowval[k], j] = csc.nzval[k]
        end
    end
    lnk
end

function findindex(lnk::SparseMatrixLNKDict, i, j)
    if !((1 <= i <= lnk.m) & (1 <= j <= lnk.n))
        throw(BoundsError(lnk, (i, j)))
    end

    k = get(lnk.colstart, j, 0)
    if k==0
        return 0,0
    end
    k0 = k
    while k > 0
        if lnk.rowval[k] == i
            return k, 0
        end
        k0 = k
        k = lnk.colptr[k]
    end
    return 0, k0
end

"""
$(SIGNATURES)
    
Return value stored for entry or zero if not found
"""
function Base.getindex(lnk::SparseMatrixLNKDict{Tv, Ti}, i, j) where {Tv, Ti}
    k, k0 = findindex(lnk, i, j)
    if k == 0
        return zero(Tv)
    else
        return lnk.nzval[k]
    end
end

function addentry!(lnk::SparseMatrixLNKDict, i, j, k, k0)
    # increase number of entries
    lnk.nentries += 1
    if length(lnk.nzval) < lnk.nentries
        newsize = Int(ceil(5.0 * lnk.nentries / 4.0))
        resize!(lnk.nzval, newsize)
        resize!(lnk.rowval, newsize)
        resize!(lnk.colptr, newsize)
    end
    
    if k0==0
        lnk.colstart[j]=lnk.nentries
    end

    # Append entry if not found
    lnk.rowval[lnk.nentries] = i

    # Shift the end of the list
    lnk.colptr[lnk.nentries] = 0

    if k0>0
        lnk.colptr[k0] = lnk.nentries
    end
    
    # Update number of nonzero entries
    lnk.nnz += 1
    return lnk.nentries
end

"""
$(SIGNATURES)
    
Update value of existing entry, otherwise extend matrix if v is nonzero.
"""
function Base.setindex!(lnk::SparseMatrixLNKDict, v, i, j)
    if !((1 <= i <= lnk.m) & (1 <= j <= lnk.n))
        throw(BoundsError(lnk, (i, j)))
    end

    k, k0 = findindex(lnk, i, j)
    if k > 0
        lnk.nzval[k] = v
        return lnk
    end
    if !iszero(v)
        k = addentry!(lnk, i, j, k, k0)
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
function updateindex!(lnk::SparseMatrixLNKDict{Tv, Ti}, op, v, i, j) where {Tv, Ti}
    k, k0 = findindex(lnk, i, j)
    if k > 0
        lnk.nzval[k] = op(lnk.nzval[k], v)
        return lnk
    end
    if !iszero(v)
        k = addentry!(lnk, i, j, k, k0)
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
function rawupdateindex!(lnk::SparseMatrixLNKDict{Tv, Ti}, op, v, i, j) where {Tv, Ti}
    k, k0 = findindex(lnk, i, j)
    if k > 0
        lnk.nzval[k] = op(lnk.nzval[k], v)
    else
        k = addentry!(lnk, i, j, k, k0)
        lnk.nzval[k] = op(zero(Tv), v)
    end
    lnk
end

"""
$(SIGNATURES)

Return tuple containing size of the matrix.
"""
Base.size(lnk::SparseMatrixLNKDict) = (lnk.m, lnk.n)

"""
$(SIGNATURES)

Return number of nonzero entries.
"""
SparseArrays.nnz(lnk::SparseMatrixLNKDict) = lnk.nnz

"""
$(SIGNATURES)

Dummy flush! method for SparseMatrixLNKDict. Just
used in test methods
"""
function flush!(lnk::SparseMatrixLNKDict{Tv, Ti}) where {Tv, Ti}
    return lnk
end

"""
    $(SIGNATURES)
Add lnk and csc via interim COO (coordinate) format, i.e. arrays I,J,V.
"""
function add_via_COO(lnk::SparseMatrixLNKDict{Tv, Ti},
                     csc::SparseMatrixCSC)::SparseMatrixCSC where {Tv, Ti <: Integer}
    (;colptr,nzval,rowval,m,n)=csc
    l=nnz(lnk)+nnz(csc)
    I=Vector{Ti}(undef,l)
    J=Vector{Ti}(undef,l)
    V=Vector{Tv}(undef,l)
    i=1
    for icsc=1:length(colptr)-1
        for j=colptr[icsc]:colptr[icsc+1]-1
            I[i]=icsc
            J[i]=rowval[j]
            V[i]=nzval[j]
            i=i+1
        end            
    end
    for (j,k) in lnk.colstart
        while k>0
            I[i]=lnk.rowval[k]
            J[i]=j
            V[i]=lnk.nzval[k]
            k=lnk.colptr[k]
            i=i+1
        end
    end
    return SparseArrays.sparse!(I,J,V,m,n,+)
end


"""
    $(SIGNATURES)
Add lnk and csc without creation of intermediate data.
"""
function add_directly(lnk::SparseMatrixLNKDict{Tv, Ti},
                      csc::SparseMatrixCSC)::SparseMatrixCSC where {Tv, Ti <: Integer}
    @assert(csc.m==lnk.m)
    @assert(csc.n==lnk.n)

    # overallocate arrays in order to avoid
    # presumably slower push!
    xnnz = nnz(csc) + nnz(lnk)
    colptr = Vector{Ti}(undef, csc.n + 1)
    rowval = Vector{Ti}(undef, xnnz)
    nzval = Vector{Tv}(undef, xnnz)

    # Detect the maximum column length of lnk
    lnk_maxcol = 0
    for (j,k) in lnk.colstart
        lcol = zero(Ti)
        while k > 0
            lcol += 1
            k = lnk.colptr[k]
        end
        lnk_maxcol = max(lcol, lnk_maxcol)
    end

    # pre-allocate column  data
    col = [ColEntry{Tv, Ti}(0, zero(Tv)) for i = 1:lnk_maxcol]

    inz = 1 # counts the nonzero entries in the new matrix

    in_csc_col(jcsc, j) = (nnz(csc) > zero(Ti)) && (jcsc < csc.colptr[j + 1])

    in_lnk_col(jlnk, l_lnk_col) = (jlnk <= l_lnk_col)

    # loop over all columns
    for j = 1:(csc.n)
        # Copy extension entries into col and sort them
        k = get(lnk.colstart, j, 0)
        l_lnk_col = 0
        while k > 0
            if lnk.rowval[k] > 0
                l_lnk_col += 1
                col[l_lnk_col] = ColEntry(lnk.rowval[k], lnk.nzval[k])
            end
            k = lnk.colptr[k]
        end
        sort!(col, 1, l_lnk_col, Base.QuickSort, Base.Forward)

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
    resize!(rowval, inz - 1)
    resize!(nzval, inz - 1)
    SparseMatrixCSC{Tv, Ti}(csc.m, csc.n, colptr, rowval, nzval)
end



"""
    $(SIGNATURES)

Add SparseMatrixCSC matrix and [`SparseMatrixLNKDict`](@ref)  lnk, returning a SparseMatrixCSC
"""
Base.:+(lnk::SparseMatrixLNKDict, csc::SparseMatrixCSC) = add_directly(lnk, csc)

function sum!(nodeparts, lnkdictmatrices::Vector{SparseMatrixLNKDict{Tv,Ti}}, cscmatrix::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    lnew=sum(nnz,lnkdictmatrices)
    if lnew>0
        (;colptr,nzval,rowval,m,n)=cscmatrix
        l=lnew+nnz(cscmatrix)
        I=Vector{Ti}(undef,l)
        J=Vector{Ti}(undef,l)
        V=Vector{Tv}(undef,l)
        i=1
        
        for icsc=1:length(colptr)-1
            for j=colptr[icsc]:colptr[icsc+1]-1
                I[i]=icsc
                J[i]=rowval[j]
                V[i]=nzval[j]
                i=i+1
            end            
        end

        ip=1
        for lnk in lnkdictmatrices
            for (j,k) in lnk.colstart
                nodeparts[j]=ip
                while k>0
                    I[i]=lnk.rowval[k]
                    J[i]=j
                    V[i]=lnk.nzval[k]
                    k=lnk.colptr[k]
                    i=i+1
                end
            end
            ip=ip+1
        end
        return SparseArrays.sparse!(I,J,V,m,n,+)
    end
    return cscmatrix
end
        


"""
$(SIGNATURES)
    
Constructor from SparseMatrixLNKDict.

"""
function SparseArrays.SparseMatrixCSC(lnk::SparseMatrixLNKDict)::SparseMatrixCSC
    csc = spzeros(lnk.m, lnk.n)
    lnk + csc
end

function SparseArrays.sparse(lnk::SparseMatrixLNKDict)
    lnk + spzeros(lnk.m, lnk.n)
end

function Base.copy(S::SparseMatrixLNKDict)
    SparseMatrixLNKDict(size(S, 1),
                        size(S, 2),
                        S.nnz,
                        S.nentries,
                        copy(S.colptr),
                        copy(S.colstart),
                        copy(S.rowvals),
                        copy(S.nzval))
end
