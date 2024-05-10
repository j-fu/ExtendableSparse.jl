##################################################################
"""
$(TYPEDEF)

Extendable sparse matrix. A nonzero  entry of this matrix is contained
either in cscmatrix, or in lnkmatrix, never in both.

$(TYPEDFIELDS)
"""
mutable struct ExtendableSparseMatrix{Tv, Ti <: Integer} <: AbstractSparseMatrixCSC{Tv, Ti}
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
    return ExtendableSparseMatrix{Tv, Ti}(csc, nothing, Base.ReentrantLock(), phash(csc))
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

 Create SparseMatrixCSC from ExtendableSparseMatrix
"""
function SparseArrays.SparseMatrixCSC(A::ExtendableSparseMatrix)
    flush!(A)
    A.cscmatrix
end


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
        lock(ext.lock)
        try
            if ext.lnkmatrix == nothing
                ext.lnkmatrix = SparseMatrixLNK{Tv, Ti}(ext.cscmatrix.m, ext.cscmatrix.n)
            end
            updateindex!(ext.lnkmatrix, op, v, i, j)
        finally
            unlock(ext.lock)
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
        lock(ext.lock)
        try
            if ext.lnkmatrix == nothing
                ext.lnkmatrix = SparseMatrixLNK{Tv, Ti}(ext.cscmatrix.m, ext.cscmatrix.n)
            end
            rawupdateindex!(ext.lnkmatrix, op, v, i, j)
        finally
            unlock(ext.lock)
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
        lock(ext.lock)
        try
            if ext.lnkmatrix == nothing
                ext.lnkmatrix = SparseMatrixLNK{Tv, Ti}(ext.cscmatrix.m, ext.cscmatrix.n)
            end
            ext.lnkmatrix[i, j] = v
        finally
            unlock(ext.lock)
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
        lock(ext.lock)
        try
            v=ext.lnkmatrix[i, j]
        finally
            unlock(ext.lock)
        end
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
function Base.show(io::IO, ::MIME"text/plain", ext::ExtendableSparseMatrix)
    flush!(ext)
    xnnz = nnz(ext)
    m, n = size(ext)
    print(io,
          m,
          "×",
          n,
          " ",
          typeof(ext),
          " with ",
          xnnz,
          " stored ",
          xnnz == 1 ? "entry" : "entries")

    if !haskey(io, :compact)
        io = IOContext(io, :compact => true)
    end

    if !(m == 0 || n == 0 || xnnz == 0)
        print(io, ":\n")
        Base.print_array(IOContext(io), ext.cscmatrix)
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

"""
$(SIGNATURES)

[`flush!`](@ref) and return number of nonzeros in ext.cscmatrix.
"""
function SparseArrays.nnz(ext::ExtendableSparseMatrix)
    flush!(ext)
    return nnz(ext.cscmatrix)
end

"""
$(SIGNATURES)

[`flush!`](@ref) and return nonzeros in ext.cscmatrix.
"""
function SparseArrays.nonzeros(ext::ExtendableSparseMatrix)
    flush!(ext)
    return nonzeros(ext.cscmatrix)
end

"""
$(SIGNATURES)

Return element type.
"""
Base.eltype(::ExtendableSparseMatrix{Tv, Ti}) where {Tv, Ti} = Tv

"""
$(SIGNATURES)

[`flush!`](@ref) and return rowvals in ext.cscmatrix.
"""
function SparseArrays.rowvals(ext::ExtendableSparseMatrix)
    flush!(ext)
    rowvals(ext.cscmatrix)
end

"""
$(SIGNATURES)

[`flush!`](@ref) and return colptr of  in ext.cscmatrix.
"""
function SparseArrays.getcolptr(ext::ExtendableSparseMatrix)
    flush!(ext)
    return getcolptr(ext.cscmatrix)
end

"""
$(SIGNATURES)

[`flush!`](@ref) and return findnz(ext.cscmatrix).
"""
function SparseArrays.findnz(ext::ExtendableSparseMatrix)
    flush!(ext)
    return findnz(ext.cscmatrix)
end

@static if VERSION >= v"1.7"
    function SparseArrays._checkbuffers(ext::ExtendableSparseMatrix)
        flush!(ext)
        SparseArrays._checkbuffers(ext.cscmatrix)
    end
end

"""
    A\b

[`\\`](@ref) for ExtendableSparse. It calls the LU factorization form Sparspak.jl, unless GPL components
are allowed  in the Julia sysimage and the floating point type of the matrix is Float64 or Complex64.
In that case, Julias standard `\` is called, which is realized via UMFPACK.
"""
function LinearAlgebra.:\(ext::ExtendableSparseMatrix{Tv, Ti},
                          b::AbstractVector) where {Tv, Ti}
    flush!(ext)
    SparspakLU(ext) \ b
end

"""
$(SIGNATURES)

[`\\`](@ref) for Symmetric{ExtendableSparse}
"""
function LinearAlgebra.:\(symm_ext::Symmetric{Tm, ExtendableSparseMatrix{Tm, Ti}},
                          b::AbstractVector) where {Tm, Ti}
    symm_ext.data \ b # no ldlt yet ...
end

"""
$(SIGNATURES)

[`\\`](@ref) for Hermitian{ExtendableSparse}
"""
function LinearAlgebra.:\(symm_ext::Hermitian{Tm, ExtendableSparseMatrix{Tm, Ti}},
                          b::AbstractVector) where {Tm, Ti}
    symm_ext.data \ B # no ldlt yet ...
end

if USE_GPL_LIBS
    for (Tv) in (:Float64, :ComplexF64)
        @eval begin function LinearAlgebra.:\(ext::ExtendableSparseMatrix{$Tv, Ti},
                                              B::AbstractVector) where {Ti}
            flush!(ext)
            ext.cscmatrix \ B
        end end

        @eval begin function LinearAlgebra.:\(symm_ext::Symmetric{$Tv,
                                                                  ExtendableSparseMatrix{
                                                                                         $Tv,
                                                                                         Ti
                                                                                         }},
                                              B::AbstractVector) where {Ti}
            flush!(symm_ext.data)
            symm_csc = Symmetric(symm_ext.data.cscmatrix, Symbol(symm_ext.uplo))
            symm_csc \ B
        end end

        @eval begin function LinearAlgebra.:\(symm_ext::Hermitian{$Tv,
                                                                  ExtendableSparseMatrix{
                                                                                         $Tv,
                                                                                         Ti
                                                                                         }},
                                              B::AbstractVector) where {Ti}
            flush!(symm_ext.data)
            symm_csc = Hermitian(symm_ext.data.cscmatrix, Symbol(symm_ext.uplo))
            symm_csc \ B
        end end
    end
end # USE_GPL_LIBS

"""
$(SIGNATURES)

[`flush!`](@ref) and ldiv with ext.cscmatrix
"""
function LinearAlgebra.ldiv!(r, ext::ExtendableSparse.ExtendableSparseMatrix, x)
    flush!(ext)
    return LinearAlgebra.ldiv!(r, ext.cscmatrix, x)
end

"""
$(SIGNATURES)

[`flush!`](@ref) and multiply with ext.cscmatrix
"""
function LinearAlgebra.mul!(r, ext::ExtendableSparse.ExtendableSparseMatrix, x)
    flush!(ext)
    return LinearAlgebra.mul!(r, ext.cscmatrix, x)
end

"""
$(SIGNATURES)

[`flush!`](@ref) and calculate norm from cscmatrix
"""
function LinearAlgebra.norm(A::ExtendableSparseMatrix, p::Real = 2)
    flush!(A)
    return LinearAlgebra.norm(A.cscmatrix, p)
end

"""
$(SIGNATURES)

[`flush!`](@ref) and calculate opnorm from cscmatrix
"""
function LinearAlgebra.opnorm(A::ExtendableSparseMatrix, p::Real = 2)
    flush!(A)
    return LinearAlgebra.opnorm(A.cscmatrix, p)
end

"""
$(SIGNATURES)

[`flush!`](@ref) and calculate cond from cscmatrix
"""
function LinearAlgebra.cond(A::ExtendableSparseMatrix, p::Real = 2)
    flush!(A)
    return LinearAlgebra.cond(A.cscmatrix, p)
end

"""
$(SIGNATURES)

[`flush!`](@ref) and check for symmetry of cscmatrix
"""
function LinearAlgebra.issymmetric(A::ExtendableSparseMatrix)
    flush!(A)
    return LinearAlgebra.issymmetric(A.cscmatrix)
end

"""
$(SIGNATURES)

Add SparseMatrixCSC matrix and [`ExtendableSparseMatrix`](@ref)  ext.
"""
function Base.:+(ext::ExtendableSparseMatrix, csc::SparseMatrixCSC)
    flush!(ext)
    return ext.cscmatrix + csc
end

function Base.:+(A::ExtendableSparseMatrix, B::ExtendableSparseMatrix)
    flush!(A)
    flush!(B)
    return ExtendableSparseMatrix(A.cscmatrix + B.cscmatrix)
end

function Base.:-(A::ExtendableSparseMatrix, B::ExtendableSparseMatrix)
    flush!(A)
    flush!(B)
    return ExtendableSparseMatrix(A.cscmatrix - B.cscmatrix)
end

function Base.:*(A::ExtendableSparseMatrix, B::ExtendableSparseMatrix)
    flush!(A)
    flush!(B)
    return ExtendableSparseMatrix(A.cscmatrix * B.cscmatrix)
end

"""
$(SIGNATURES)
"""
function Base.:*(d::Diagonal, ext::ExtendableSparseMatrix)
    flush!(ext)
    return ExtendableSparseMatrix(d * ext.cscmatrix)
end

"""
$(SIGNATURES)
"""
function Base.:*(ext::ExtendableSparseMatrix, d::Diagonal)
    flush!(ext)
    return ExtendableSparseMatrix(ext.cscmatrix * d)
end

"""
$(SIGNATURES)

Subtract  SparseMatrixCSC matrix from  [`ExtendableSparseMatrix`](@ref)  ext.
"""
function Base.:-(ext::ExtendableSparseMatrix, csc::SparseMatrixCSC)
    flush!(ext)
    return ext.cscmatrix - csc
end

"""
$(SIGNATURES)

Subtract  [`ExtendableSparseMatrix`](@ref)  ext from  SparseMatrixCSC.
"""
function Base.:-(csc::SparseMatrixCSC, ext::ExtendableSparseMatrix)
    flush!(ext)
    return csc - ext.cscmatrix
end

"""
$(SIGNATURES)
"""
function SparseArrays.dropzeros!(ext::ExtendableSparseMatrix)
    flush!(ext)
    dropzeros!(ext.cscmatrix)
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


function mark_dirichlet(A::ExtendableSparseMatrix;penalty=1.0e20)
    flush!(A)
    mark_dirichlet(A.cscmatrix;penalty)
end

function eliminate_dirichlet(A::ExtendableSparseMatrix,dirichlet)
    flush!(A)
    ExtendableSparseMatrix(eliminate_dirichlet(A.cscmatrix,dirichlet))
end

function eliminate_dirichlet!(A::ExtendableSparseMatrix,dirichlet)
    flush!(A)
    eliminate_dirichlet!(A.cscmatrix,dirichlet)
    A
end

