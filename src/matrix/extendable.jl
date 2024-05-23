##################################################################
"""
$(TYPEDEF)

Extendable sparse matrix. A nonzero  entry of this matrix is contained
either in cscmatrix, or in lnkmatrix, never in both.

$(TYPEDFIELDS)
"""
mutable struct ExtendableSparseMatrix{Tv, Ti <: Integer} <: AbstractExtendableSparseMatrix{Tv, Ti}
    """
    Final matrix data
    """
    cscmatrix::SparseMatrixCSC{Tv, Ti}

    """
    Linked list structure holding data of extension
    """
    lnkmatrix::Union{SparseMatrixLNK{Tv, Ti}, Nothing}

    lock::Base.ReentrantLock
    
    """
    Pattern hash
    """
    phash::UInt64
end

mutable struct Locking
    locking::Bool
end

#
# Locking functionality just for developing parallelization.
# To be removed before merging into main branch.
#
const locking=Locking(false)

function with_locking!(l::Bool)
    global locking
    locking.locking=l
end

function with_locking()
    global locking
    locking.locking
end

mylock(x)=with_locking() ? Base.lock(x) : nothing
myunlock(x)=with_locking() ? Base.unlock(x) : nothing


#mylock(x)=nothing
#myunlock(x)=nothing

"""
```
ExtendableSparseMatrix(Tv,Ti,m,n)
ExtendableSparseMatrix(Tv,m,n)
ExtendableSparseMatrix(m,n)
```
Create empty ExtendableSparseMatrix. This is equivalent to `spzeros(m,n)` for
`SparseMartrixCSC`.

"""

function ExtendableSparseMatrix{Tv, Ti}(m, n) where {Tv, Ti <: Integer}
    ExtendableSparseMatrix{Tv, Ti}(spzeros(Tv, Ti, m, n), nothing,Base.ReentrantLock(), 0)
end

function ExtendableSparseMatrix(valuetype::Type{Tv},
                                indextype::Type{Ti},
                                m,
                                n) where {Tv, Ti <: Integer}
    ExtendableSparseMatrix{Tv, Ti}(m, n)
end

function ExtendableSparseMatrix(valuetype::Type{Tv}, m, n) where {Tv}
    ExtendableSparseMatrix{Tv, Int}(m, n)
end

ExtendableSparseMatrix(m, n) = ExtendableSparseMatrix{Float64, Int}(m, n)

"""
$(SIGNATURES)

Create ExtendableSparseMatrix from SparseMatrixCSC
"""
function ExtendableSparseMatrix(csc::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti <: Integer}
    ExtendableSparseMatrix{Tv, Ti}(csc, nothing, Base.ReentrantLock(), phash(csc))
end

function ExtendableSparseMatrix{Tv,Ti}(csc::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti <: Integer}
    ExtendableSparseMatrix{Tv, Ti}(csc, nothing, Base.ReentrantLock(), phash(csc))
end

"""
$(SIGNATURES)

 Create ExtendableSparseMatrix from Diagonal
"""
ExtendableSparseMatrix(D::Diagonal) = ExtendableSparseMatrix(sparse(D))

"""
$(SIGNATURES)

 Create ExtendableSparseMatrix from AbstractMatrix, dropping all zero entries.
 This is the equivalent to `sparse(A)`.
"""
ExtendableSparseMatrix(A::AbstractMatrix) = ExtendableSparseMatrix(sparse(A))

"""
    ExtendableSparseMatrix(I,J,V)
    ExtendableSparseMatrix(I,J,V,m,n)
    ExtendableSparseMatrix(I,J,V,combine::Function)
    ExtendableSparseMatrix(I,J,V,m,n,combine::Function)

Create ExtendableSparseMatrix from triplet (COO) data.
"""
ExtendableSparseMatrix(I, J, V::AbstractVector) = ExtendableSparseMatrix(sparse(I, J, V))

function ExtendableSparseMatrix(I, J, V::AbstractVector, m, n)
    ExtendableSparseMatrix(sparse(I, J, V, m, n))
end

function ExtendableSparseMatrix(I, J, V::AbstractVector, combine::Function)
    ExtendableSparseMatrix(sparse(I, J, V, combine))
end

function ExtendableSparseMatrix(I, J, V::AbstractVector, m, n, combine::Function)
    ExtendableSparseMatrix(sparse(I, J, V, m, n, combine))
end

# THese are probably too much...
# function Base.transpose(A::ExtendableSparseMatrix)
#     flush!(A)
#     ExtendableSparseMatrix(Base.transpose(sparse(A)))
# end
# function Base.adjoint(A::ExtendableSparseMatrix)
#     flush!(A)
#     ExtendableSparseMatrix(Base.adjoint(sparse(A)))
# end
# function SparseArrays.sparse(text::LinearAlgebra.Transpose{Tv,ExtendableSparseMatrix{Tv,Ti}}) where {Tv,Ti}
#     transpose(sparse(parent(text)))
# end



"""
$(SIGNATURES)

Create similar but emtpy extendableSparseMatrix
"""
function Base.similar(m::ExtendableSparseMatrix{Tv, Ti}) where {Tv, Ti}
    ExtendableSparseMatrix{Tv, Ti}(size(m)...)
end

function Base.similar(m::ExtendableSparseMatrix{Tv, Ti}, ::Type{T}) where {Tv, Ti, T}
    ExtendableSparseMatrix{T, Ti}(size(m)...)
end

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

function updateindex!(ext::ExtendableSparseMatrix{Tv, Ti},
                      op,
                      v,
                      i,
                      j) where {Tv, Ti <: Integer}
    k = findindex(ext.cscmatrix, i, j)
    if k > 0
        ext.cscmatrix.nzval[k] = op(ext.cscmatrix.nzval[k], v)
    else
        mylock(ext.lock)
        try
            if ext.lnkmatrix == nothing
                ext.lnkmatrix = SparseMatrixLNK{Tv, Ti}(ext.cscmatrix.m, ext.cscmatrix.n)
            end
            updateindex!(ext.lnkmatrix, op, v, i, j)
        finally
            myunlock(ext.lock)
        end
    end
    ext
end

"""
$(SIGNATURES)
Like [`updateindex!`](@ref) but without 
checking if v is zero.
"""
function rawupdateindex!(ext::ExtendableSparseMatrix{Tv, Ti},
                         op,
                         v,
                         i,
                         j) where {Tv, Ti <: Integer}
    k = findindex(ext.cscmatrix, i, j)
    if k > 0
        ext.cscmatrix.nzval[k] = op(ext.cscmatrix.nzval[k], v)
    else
        mylock(ext.lock)
        try
            if ext.lnkmatrix == nothing
                ext.lnkmatrix = SparseMatrixLNK{Tv, Ti}(ext.cscmatrix.m, ext.cscmatrix.n)
            end
            rawupdateindex!(ext.lnkmatrix, op, v, i, j)
        finally
            myunlock(ext.lock)
        end
    end
    ext
end

"""
$(SIGNATURES)

Find index in CSC matrix and set value if it exists. Otherwise,
set index in extension if `v` is nonzero.
"""
function Base.setindex!(ext::ExtendableSparseMatrix{Tv, Ti},
                        v::Union{Number,AbstractVecOrMat},
                        i::Integer,
                        j::Integer) where {Tv, Ti}
    k = findindex(ext.cscmatrix, i, j)
    if k > 0
        ext.cscmatrix.nzval[k] = v
    else
        mylock(ext.lock)
        try
            if ext.lnkmatrix == nothing
                ext.lnkmatrix = SparseMatrixLNK{Tv, Ti}(ext.cscmatrix.m, ext.cscmatrix.n)
            end
            ext.lnkmatrix[i, j] = v
        finally
            myunlock(ext.lock)
        end
    end
end

"""
$(SIGNATURES)

Find index in CSC matrix and return value, if it exists.
Otherwise, return value from extension.
"""
function Base.getindex(ext::ExtendableSparseMatrix{Tv, Ti},
                       i::Integer,
                       j::Integer) where {Tv, Ti <: Integer}
    k = findindex(ext.cscmatrix, i, j)
    if k > 0
        return ext.cscmatrix.nzval[k]
    elseif ext.lnkmatrix == nothing
        return zero(Tv)
    else
        v=zero(Tv)
        mylock(ext.lock)
        try
            v=ext.lnkmatrix[i, j]
        finally
            myunlock(ext.lock)
        end
    end
end



"""
$(SIGNATURES)

If there are new entries in extension, create new CSC matrix by adding the
cscmatrix and linked list matrix and reset the linked list based extension.
"""
function flush!(ext::ExtendableSparseMatrix)
    if ext.lnkmatrix != nothing && nnz(ext.lnkmatrix) > 0
        ext.cscmatrix = ext.lnkmatrix + ext.cscmatrix
        ext.lnkmatrix = nothing
        ext.phash = phash(ext.cscmatrix)
    end
    return ext
end


function SparseArrays.sparse(ext::ExtendableSparseMatrix)
    flush!(ext)
    ext.cscmatrix
end


"""
$(SIGNATURES)

Reset ExtenableSparseMatrix into state similar to that after creation.
"""
function reset!(A::ExtendableSparseMatrix)
    A.cscmatrix=spzeros(size(A)...)
    A.lnkmatrix=nothing
end



"""
$(SIGNATURES)
"""
function Base.copy(S::ExtendableSparseMatrix)
    if isnothing(S.lnkmatrix)
        ExtendableSparseMatrix(copy(S.cscmatrix), nothing,  Base.ReentrantLock(),S.phash)
    else
        ExtendableSparseMatrix(copy(S.cscmatrix), copy(S.lnkmatrix), Base.ReentrantLock(), S.phash)
    end
end

"""
    pointblock(matrix,blocksize)

Create a pointblock matrix.
"""
function pointblock(A0::ExtendableSparseMatrix{Tv,Ti},blocksize) where {Tv,Ti}
    A=SparseMatrixCSC(A0)
    colptr=A.colptr
    rowval=A.rowval
    nzval=A.nzval
    n=A.n
    block=zeros(Tv,blocksize,blocksize)
    nblock=n÷blocksize
    b=SMatrix{blocksize,blocksize}(block)
    Tb=typeof(b)
    Ab=ExtendableSparseMatrix{Tb,Ti}(nblock,nblock)
    
    
    for i=1:n
	for k=colptr[i]:colptr[i+1]-1
	    j=rowval[k]
	    iblock=(i-1)÷blocksize+1
	    jblock=(j-1)÷blocksize+1
	    ii=(i-1)%blocksize+1
	    jj=(j-1)%blocksize+1
	    block[ii,jj]=nzval[k]
	    rawupdateindex!(Ab,+,SMatrix{blocksize,blocksize}(block),iblock,jblock)
	    block[ii,jj]=zero(Tv)
	end
    end
    flush!(Ab)
end


