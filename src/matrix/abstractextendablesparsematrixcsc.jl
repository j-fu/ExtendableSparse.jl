"""

Subtypes must implement:
- SparseArrays.sparse (may be should be sparse! ?) flush+return SparseMatrixCSC
- Constructor from SparseMatrixCSC
rawupdateindex!
reset!: empty all internals, just keep size 
"""

abstract type AbstractExtendableSparseMatrixCSC{Tv,Ti} <: AbstractSparseMatrixCSC{Tv,Ti} end

"""
$(SIGNATURES)

[`flush!`](@ref) and return number of nonzeros in ext.cscmatrix.
"""
SparseArrays.nnz(ext::AbstractExtendableSparseMatrixCSC)=nnz(sparse(ext))

"""
$(SIGNATURES)

[`flush!`](@ref) and return nonzeros in ext.cscmatrix.
"""
SparseArrays.nonzeros(ext::AbstractExtendableSparseMatrixCSC)=nonzeros(sparse(ext))

Base.size(ext::AbstractExtendableSparseMatrixCSC)=size(ext.cscmatrix)



"""
$(SIGNATURES)

Return element type.
"""
Base.eltype(::AbstractExtendableSparseMatrixCSC{Tv, Ti}) where {Tv, Ti} = Tv



"""
$(SIGNATURES)

 Create SparseMatrixCSC from ExtendableSparseMatrix
"""
SparseArrays.SparseMatrixCSC(A::AbstractExtendableSparseMatrixCSC)=sparse(A)




function Base.show(io::IO, ::MIME"text/plain", ext::AbstractExtendableSparseMatrixCSC)
    A=sparse(ext)
    xnnz = nnz(A)
    m, n = size(A)
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
        Base.print_array(IOContext(io), A)
    end
end


"""
$(SIGNATURES)

[`flush!`](@ref) and return rowvals in ext.cscmatrix.
"""
SparseArrays.rowvals(ext::AbstractExtendableSparseMatrixCSC)=rowvals(sparse(ext))


"""
$(SIGNATURES)

[`flush!`](@ref) and return colptr of  in ext.cscmatrix.
"""
SparseArrays.getcolptr(ext::AbstractExtendableSparseMatrixCSC)=getcolptr(sparse(ext))

    
"""
$(SIGNATURES)

[`flush!`](@ref) and return findnz(ext.cscmatrix).
"""
SparseArrays.findnz(ext::AbstractExtendableSparseMatrixCSC)=findnz(sparse(ext))


@static if VERSION >= v"1.7"
    SparseArrays._checkbuffers(ext::AbstractExtendableSparseMatrixCSC)=  SparseArrays._checkbuffers(sparse(ext))
end

"""
    A\b

[`\\`](@ref) for ExtendableSparse. It calls the LU factorization form Sparspak.jl, unless GPL components
are allowed  in the Julia sysimage and the floating point type of the matrix is Float64 or Complex64.
In that case, Julias standard `\` is called, which is realized via UMFPACK.
"""
function LinearAlgebra.:\(ext::AbstractExtendableSparseMatrixCSC{Tv, Ti},
                          b::AbstractVector) where {Tv, Ti}
    SparspakLU(sparse(ext)) \ b
end


"""
$(SIGNATURES)

[`\\`](@ref) for Symmetric{ExtendableSparse}
"""
function LinearAlgebra.:\(symm_ext::Symmetric{Tm, T},
                          b::AbstractVector) where {Tm, Ti, T<:AbstractExtendableSparseMatrixCSC{Tm,Ti}}
    Symmetric(sparse(symm_ext.data),Symbol(symm_ext.uplo)) \ b # no ldlt yet ...
end

"""
$(SIGNATURES)

[`\\`](@ref) for Hermitian{ExtendableSparse}
"""
function LinearAlgebra.:\(symm_ext::Hermitian{Tm, T},
                          b::AbstractVector) where {Tm, Ti, T<:AbstractExtendableSparseMatrixCSC{Tm,Ti}}
    Hermitian(sparse(symm_ext.data),Symbol(symm_ext.uplo)) \ b # no ldlt yet ...
end

if USE_GPL_LIBS
    for (Tv) in (:Float64, :ComplexF64)
        @eval begin function LinearAlgebra.:\(ext::AbstractExtendableSparseMatrixCSC{$Tv, Ti},
                                              B::AbstractVector) where {Ti}
            sparse(ext) \ B
        end end

        @eval begin function LinearAlgebra.:\(symm_ext::Symmetric{$Tv,
                                                                  AbstractExtendableSparseMatrixCSC{
                                                                      $Tv,
                                                                      Ti
                                                                  }},
            B::AbstractVector) where {Ti}
            symm_csc = Symmetric(sparse(symm_ext.data), Symbol(symm_ext.uplo))
            symm_csc \ B
        end end

        @eval begin function LinearAlgebra.:\(symm_ext::Hermitian{$Tv,
                                                                  AbstractExtendableSparseMatrixCSC{
                                                                      $Tv,
                                                                      Ti
                                                                  }},
                                              B::AbstractVector) where {Ti}
            symm_csc = Hermitian(sparse(symm_ext.data), Symbol(symm_ext.uplo))
            symm_csc \ B
        end end
    end
end # USE_GPL_LIBS

"""
$(SIGNATURES)

[`flush!`](@ref) and ldiv with ext.cscmatrix
"""
function LinearAlgebra.ldiv!(r, ext::AbstractExtendableSparseMatrixCSC, x)
    LinearAlgebra.ldiv!(r, sparse(ext), x)
end

"""
$(SIGNATURES)

[`flush!`](@ref) and multiply with ext.cscmatrix
"""
function LinearAlgebra.mul!(r, ext::AbstractExtendableSparseMatrixCSC, x)
    LinearAlgebra.mul!(r, sparse(ext), x)
end

"""
$(SIGNATURES)

[`flush!`](@ref) and calculate norm from cscmatrix
"""
function LinearAlgebra.norm(A::AbstractExtendableSparseMatrixCSC, p::Real = 2)
    return LinearAlgebra.norm(sparse(A), p)
end

"""
$(SIGNATURES)

[`flush!`](@ref) and calculate opnorm from cscmatrix
"""
function LinearAlgebra.opnorm(A::AbstractExtendableSparseMatrixCSC, p::Real = 2)
    return LinearAlgebra.opnorm(sparse(A), p)
end

"""
$(SIGNATURES)

[`flush!`](@ref) and calculate cond from cscmatrix
"""
function LinearAlgebra.cond(A::AbstractExtendableSparseMatrixCSC, p::Real = 2)
    return LinearAlgebra.cond(sparse(A), p)
end

"""
$(SIGNATURES)

[`flush!`](@ref) and check for symmetry of cscmatrix
"""
function LinearAlgebra.issymmetric(A::AbstractExtendableSparseMatrixCSC)
    return LinearAlgebra.issymmetric(sparse(A))
end
    
    


    

function Base.:+(A::T, B::T) where T<:AbstractExtendableSparseMatrixCSC
    T(sparse(A) + sparse(B))
end

function Base.:-(A::T, B::T) where T<:AbstractExtendableSparseMatrixCSC
    T(sparse(A) - sparse(B))
end

function Base.:*(A::T, B::T) where T<:AbstractExtendableSparseMatrixCSC
    T(sparse(A) * sparse(B))
end

"""
$(SIGNATURES)
"""
function Base.:*(d::Diagonal, ext::T)where T<:AbstractExtendableSparseMatrixCSC
    return T(d * sparse(ext))
end

"""
$(SIGNATURES)
"""
function Base.:*(ext::T, d::Diagonal) where  T<:AbstractExtendableSparseMatrixCSC
    return T(sparse(ext) * d)
end


"""
$(SIGNATURES)

Add SparseMatrixCSC matrix and [`ExtendableSparseMatrix`](@ref)  ext.
"""
function Base.:+(ext::AbstractExtendableSparseMatrixCSC, csc::SparseMatrixCSC)
    return sparse(ext) + csc
end


"""
$(SIGNATURES)

Subtract  SparseMatrixCSC matrix from  [`ExtendableSparseMatrix`](@ref)  ext.
"""
function Base.:-(ext::AbstractExtendableSparseMatrixCSC, csc::SparseMatrixCSC)
    return sparse(ext) - csc
end

"""
$(SIGNATURES)

Subtract  [`ExtendableSparseMatrix`](@ref)  ext from  SparseMatrixCSC.
"""
function Base.:-(csc::SparseMatrixCSC, ext::AbstractExtendableSparseMatrixCSC)
    return csc - sparse(ext)
end

"""
$(SIGNATURES)
"""
function SparseArrays.dropzeros!(ext::AbstractExtendableSparseMatrixCSC)
    dropzeros!(sparse(ext))
end



function mark_dirichlet(A::AbstractExtendableSparseMatrixCSC;penalty=1.0e20)
    mark_dirichlet(sparse(A);penalty)
end

function eliminate_dirichlet(A::T,dirichlet) where T<:AbstractExtendableSparseMatrixCSC
   T(eliminate_dirichlet(sparse(A),dirichlet))
end

function eliminate_dirichlet!(A::AbstractExtendableSparseMatrixCSC,dirichlet)
    eliminate_dirichlet!(sparse(A),dirichlet)
    A
end

