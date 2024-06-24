##################################################################
"""
$(TYPEDEF)

Extendable sparse matrix. A nonzero  entry of this matrix is contained
either in cscmatrix, or in lnkmatrix, never in both.

$(TYPEDFIELDS)
"""
mutable struct ExtendableSparseMatrixCSC{Tv, Ti <: Integer} <: AbstractExtendableSparseMatrixCSC{Tv, Ti}
    """
    Final matrix data
    """
    cscmatrix::SparseMatrixCSC{Tv, Ti}

    """
    Linked list structure holding data of extension
    """
    lnkmatrix::Union{SparseMatrixLNK{Tv, Ti}, Nothing}
    
    """
    Pattern hash
    """
    phash::UInt64
end


"""
```
ExtendableSparseMatrixCSC(Tv,Ti,m,n)
ExtendableSparseMatrixCSC(Tv,m,n)
ExtendableSparseMatrixCSC(m,n)
```
Create empty ExtendableSparseMatrixCSC. This is equivalent to `spzeros(m,n)` for
`SparseMartrixCSC`.

"""

function ExtendableSparseMatrixCSC{Tv, Ti}(m, n) where {Tv, Ti <: Integer}
    ExtendableSparseMatrixCSC{Tv, Ti}(spzeros(Tv, Ti, m, n), nothing, 0)
end

function ExtendableSparseMatrixCSC(valuetype::Type{Tv},
                                indextype::Type{Ti},
                                m,
                                n) where {Tv, Ti <: Integer}
    ExtendableSparseMatrixCSC{Tv, Ti}(m, n)
end

function ExtendableSparseMatrixCSC(valuetype::Type{Tv}, m, n) where {Tv}
    ExtendableSparseMatrixCSC{Tv, Int}(m, n)
end

ExtendableSparseMatrixCSC(m, n) = ExtendableSparseMatrixCSC{Float64, Int}(m, n)

"""
$(SIGNATURES)

Create ExtendableSparseMatrixCSC from SparseMatrixCSC
"""
function ExtendableSparseMatrixCSC(csc::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti <: Integer}
    ExtendableSparseMatrixCSC{Tv, Ti}(csc, nothing, phash(csc))
end

function ExtendableSparseMatrixCSC{Tv,Ti}(csc::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti <: Integer}
    ExtendableSparseMatrixCSC{Tv, Ti}(csc, nothing, phash(csc))
end

"""
$(SIGNATURES)

 Create ExtendableSparseMatrixCSC from Diagonal
"""
ExtendableSparseMatrixCSC(D::Diagonal) = ExtendableSparseMatrixCSC(sparse(D))

"""
$(SIGNATURES)

 Create ExtendableSparseMatrixCSC from AbstractMatrix, dropping all zero entries.
 This is the equivalent to `sparse(A)`.
"""
ExtendableSparseMatrixCSC(A::AbstractMatrix) = ExtendableSparseMatrixCSC(sparse(A))

"""
    ExtendableSparseMatrixCSC(I,J,V)
    ExtendableSparseMatrixCSC(I,J,V,m,n)
    ExtendableSparseMatrixCSC(I,J,V,combine::Function)
    ExtendableSparseMatrixCSC(I,J,V,m,n,combine::Function)

Create ExtendableSparseMatrixCSC from triplet (COO) data.
"""
ExtendableSparseMatrixCSC(I, J, V::AbstractVector) = ExtendableSparseMatrixCSC(sparse(I, J, V))

function ExtendableSparseMatrixCSC(I, J, V::AbstractVector, m, n)
    ExtendableSparseMatrixCSC(sparse(I, J, V, m, n))
end

function ExtendableSparseMatrixCSC(I, J, V::AbstractVector, combine::Function)
    ExtendableSparseMatrixCSC(sparse(I, J, V, combine))
end

function ExtendableSparseMatrixCSC(I, J, V::AbstractVector, m, n, combine::Function)
    ExtendableSparseMatrixCSC(sparse(I, J, V, m, n, combine))
end

# THese are probably too much...
# function Base.transpose(A::ExtendableSparseMatrixCSC)
#     flush!(A)
#     ExtendableSparseMatrixCSC(Base.transpose(sparse(A)))
# end
# function Base.adjoint(A::ExtendableSparseMatrixCSC)
#     flush!(A)
#     ExtendableSparseMatrixCSC(Base.adjoint(sparse(A)))
# end
# function SparseArrays.sparse(text::LinearAlgebra.Transpose{Tv,ExtendableSparseMatrixCSC{Tv,Ti}}) where {Tv,Ti}
#     transpose(sparse(parent(text)))
# end



"""
$(SIGNATURES)

Create similar but emtpy extendableSparseMatrix
"""
function Base.similar(m::ExtendableSparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    ExtendableSparseMatrixCSC{Tv, Ti}(size(m)...)
end

function Base.similar(m::ExtendableSparseMatrixCSC{Tv, Ti}, ::Type{T}) where {Tv, Ti, T}
    ExtendableSparseMatrixCSC{T, Ti}(size(m)...)
end

"""
$(SIGNATURES)

Update element of the matrix  with operation `op`.
This can replace the following code and save one index
search during acces:

```@example
using ExtendableSparse # hide
A=ExtendableSparseMatrixCSC(3,3)
A[1,2]+=0.1
A
```

```@example
using ExtendableSparse # hide

A=ExtendableSparseMatrixCSC(3,3)
updateindex!(A,+,0.1,1,2)
A
```

If `v` is zero, no new entry is created.
"""

function updateindex!(ext::ExtendableSparseMatrixCSC{Tv, Ti},
                      op,
                      v,
                      i,
                      j) where {Tv, Ti <: Integer}
    k = findindex(ext.cscmatrix, i, j)
    if k > 0
        ext.cscmatrix.nzval[k] = op(ext.cscmatrix.nzval[k], v)
    else
        if ext.lnkmatrix == nothing
            ext.lnkmatrix = SparseMatrixLNK{Tv, Ti}(ext.cscmatrix.m, ext.cscmatrix.n)
        end
        updateindex!(ext.lnkmatrix, op, v, i, j)
    end
    ext
end

"""
$(SIGNATURES)
Like [`updateindex!`](@ref) but without 
checking if v is zero.
"""
function rawupdateindex!(ext::ExtendableSparseMatrixCSC{Tv, Ti},
                         op,
                         v,
                         i,
                         j) where {Tv, Ti <: Integer}
    k = findindex(ext.cscmatrix, i, j)
    if k > 0
        ext.cscmatrix.nzval[k] = op(ext.cscmatrix.nzval[k], v)
    else
            if ext.lnkmatrix == nothing
                ext.lnkmatrix = SparseMatrixLNK{Tv, Ti}(ext.cscmatrix.m, ext.cscmatrix.n)
            end
            rawupdateindex!(ext.lnkmatrix, op, v, i, j)
    end
    ext
end

"""
$(SIGNATURES)

Find index in CSC matrix and set value if it exists. Otherwise,
set index in extension if `v` is nonzero.
"""
function Base.setindex!(ext::ExtendableSparseMatrixCSC{Tv, Ti},
                        v::Union{Number,AbstractVecOrMat},
                        i::Integer,
                        j::Integer) where {Tv, Ti}
    k = findindex(ext.cscmatrix, i, j)
    if k > 0
        ext.cscmatrix.nzval[k] = v
    else
        if ext.lnkmatrix == nothing
            ext.lnkmatrix = SparseMatrixLNK{Tv, Ti}(ext.cscmatrix.m, ext.cscmatrix.n)
        end
        ext.lnkmatrix[i, j] = v
    end
end

"""
$(SIGNATURES)

Find index in CSC matrix and return value, if it exists.
Otherwise, return value from extension.
"""
function Base.getindex(ext::ExtendableSparseMatrixCSC{Tv, Ti},
                       i::Integer,
                       j::Integer) where {Tv, Ti <: Integer}
    k = findindex(ext.cscmatrix, i, j)
    if k > 0
        return ext.cscmatrix.nzval[k]
    elseif ext.lnkmatrix == nothing
        return zero(Tv)
    else
        v=zero(Tv)
        v=ext.lnkmatrix[i, j]
    end
end



"""
$(SIGNATURES)

If there are new entries in extension, create new CSC matrix by adding the
cscmatrix and linked list matrix and reset the linked list based extension.
"""
function flush!(ext::ExtendableSparseMatrixCSC)
    if ext.lnkmatrix != nothing && nnz(ext.lnkmatrix) > 0
        ext.cscmatrix = ext.lnkmatrix + ext.cscmatrix
        ext.lnkmatrix = nothing
        ext.phash = phash(ext.cscmatrix)
    end
    return ext
end


function SparseArrays.sparse(ext::ExtendableSparseMatrixCSC)
    flush!(ext)
    ext.cscmatrix
end


"""
$(SIGNATURES)

Reset ExtenableSparseMatrix into state similar to that after creation.
"""
function reset!(A::ExtendableSparseMatrixCSC)
    A.cscmatrix=spzeros(size(A)...)
    A.lnkmatrix=nothing
end



"""
$(SIGNATURES)
"""
function Base.copy(S::ExtendableSparseMatrixCSC)
    if isnothing(S.lnkmatrix)
        ExtendableSparseMatrixCSC(copy(S.cscmatrix), nothing,S.phash)
    else
        ExtendableSparseMatrixCSC(copy(S.cscmatrix), copy(S.lnkmatrix), S.phash)
    end
end

"""
    pointblock(matrix,blocksize)

Create a pointblock matrix.
"""
function pointblock(A0::ExtendableSparseMatrixCSC{Tv,Ti},blocksize) where {Tv,Ti}
    A=SparseMatrixCSC(A0)
    colptr=A.colptr
    rowval=A.rowval
    nzval=A.nzval
    n=A.n
    block=zeros(Tv,blocksize,blocksize)
    nblock=n÷blocksize
    b=SMatrix{blocksize,blocksize}(block)
    Tb=typeof(b)
    Ab=ExtendableSparseMatrixCSC{Tb,Ti}(nblock,nblock)
    
    
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


