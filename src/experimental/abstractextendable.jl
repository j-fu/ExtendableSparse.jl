abstract type AbstractExtendableSparseMatrix{Tv,Ti} <: AbstractSparseMatrixCSC{Tv,Ti} end

SparseArrays.nnz(ext::AbstractExtendableSparseMatrix)=nnz(sparse(ext))

SparseArrays.nonzeros(ext::AbstractExtendableSparseMatrix)=nonzeros(sparse(ext))

Base.size(ext::AbstractExtendableSparseMatrix)=size(sparse(ext))

function Base.show(io::IO, ::MIME"text/plain", ext::AbstractExtendableSparseMatrix)
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

    
