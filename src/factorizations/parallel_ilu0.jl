mutable struct _ParallelILU0Preconditioner{Tv,Ti}
    A::SparseMatrixCSC{Tv,Ti}
    xdiag::Vector{Tv}
    idiag::Vector{Ti}
    coloring::Vector{Vector{Ti}}
    coloring_index::Vector{Vector{Ti}}
    coloring_index_reverse::Vector{Vector{Ti}}
end

function pilu0(A0::SparseMatrixCSC{Tv,Ti}) where {Tv, Ti}
    coloring = graphcol(A0)
    coloring_index, coloring_index_reverse = coloringindex(coloring)
    A = ExtendableSparseMatrix(reordermatrix(A0, coloring)).cscmatrix

    colptr = A.colptr
    rowval = A.rowval
    nzval =  A.nzval
    n = A.n
    xdiag=Vector{Tv}(undef,n)
    idiag=Vector{Ti}(undef,n)

    # Find main diagonal index and
    # copy main diagonal values
    @inbounds for j = 1:n
        @inbounds for k = colptr[j]:(colptr[j + 1] - 1)
            i = rowval[k]
            if i == j
                idiag[j] = k
                break
            end
        end
    end
    
    @inbounds for j = 1:n
        xdiag[j] = one(Tv) / nzval[idiag[j]]
        @inbounds for k = (idiag[j] + 1):(colptr[j + 1] - 1)
            i = rowval[k]
            for l = colptr[i]:(colptr[i + 1] - 1)
                if rowval[l] == j
                    xdiag[i] -= nzval[l] * xdiag[j] * nzval[k]
                    break
                end
            end
        end
    end
    _ParallelILU0Preconditioner(A,xdiag,idiag, coloring, coloring_index,coloring_index_reverse)
end

function LinearAlgebra.ldiv!(u::AbstractVector,
                             precon::_ParallelILU0Preconditioner{Tv,Ti},
                             v::AbstractVector) where {Tv,Ti}
    A = precon.A
    colptr = A.colptr
    rowval = A.rowval
    n = A.n
    nzval = A.nzval
    xdiag = precon.xdiag
    idiag = precon.idiag

    coloring = precon.coloring
    coloring_index = precon.coloring_index
    coloring_index_reverse = precon.coloring_index_reverse
    
    Threads.@threads for j = 1:n
        u[j] = xdiag[j] * v[j]
    end

    for indset in coloring_index_reverse
        Threads.@threads for j in indset
            for k = (idiag[j] + 1):(colptr[j + 1] - 1)
                i = rowval[k]
                u[i] -= xdiag[i] * nzval[k] * u[j]
            end
        end
    end
    for indset in coloring_index
        Threads.@threads for j in indset
            for k = colptr[j]:(idiag[j] - 1)
                i = rowval[k]
                u[i] -= xdiag[i] * nzval[k] * u[j]
            end
        end
    end
end

function LinearAlgebra.ldiv!(precon::_ParallelILU0Preconditioner{Tv,Ti},
                             v::AbstractVector) where {Tv,Ti}
    ldiv!(v, precon, v)
end

# Returns an independent set of the graph of a matrix
# Reference: https://research.nvidia.com/sites/default/files/pubs/2015-05_Parallel-Graph-Coloring/nvr-2015-001.pdf
function indset(A::SparseMatrixCSC{Tv,Ti}, W::StridedVector) where {Tv, Ti}
    # Random numbers for all vertices
    lenW = length(W)
    # r = sample(1:lenW, lenW, replace = false)
    r = rand(lenW)
    @inbounds for i = 1:lenW
        if W[i] == 0
            r[i] = 0
        end
    end
    # Empty independent set
    S = zeros(Int, lenW)
    # Get independent set by comparing random number of vertex with the random 
    # numbers of all neighbor vertices
    @inbounds Threads.@threads for i = 1:lenW
        if W[i] != 0
            j = A.rowval[A.colptr[i]:(A.colptr[i + 1] - 1)]
            if all(x -> x == 1, r[i] .>= r[j])
                S[i] = W[i]
            end
        end
    end
    # Remove zero entries and return independent set
    return filter!(x -> x ≠ 0, S)
end

# Returns coloring of the graph of a matrix
# Reference: https://research.nvidia.com/sites/default/files/pubs/2015-05_Parallel-Graph-Coloring/nvr-2015-001.pdf
function graphcol(A::SparseMatrixCSC{Tv,Ti}) where {Tv, Ti}
    # Empty list for coloring
    C = Vector{Ti}[]
    # Array of vertices
    W = Ti[1:size(A)[1];]
    # Get all independent sets of the graph of the matrix
    while any(W .!= 0)
        # Get independent set
        S = indset(A + transpose(A), W)
        push!(C, S)
        # Remove entries in S from W
        @inbounds for s in S
            W[s] = 0
        end
    end
    # Return coloring
    return C
end

# Reorders a sparse matrix with provided coloring
function reordermatrix(A::SparseMatrixCSC{Tv, Ti},
                       coloring) where {Tv, Ti}
    c = collect(Iterators.flatten(coloring))
    return A[c, :][:, c]
end

# Reorders a linear system with provided coloring
function reorderlinsys(A::SparseMatrixCSC{Tv, Ti},
                       b::Vector{Tv} ,
                       coloring) where {Tv, Ti}
    c = collect(Iterators.flatten(coloring))
    return A[c, :][:, c], b[c]
end

# Returns an array with the same structure of the input coloring and ordered 
# entries 1:length(coloring) and an array with the structure of 
# reverse(coloring) and ordered entries length(coloring):-1:1 
function coloringindex(coloring)
    # First array
    c = deepcopy(coloring)
    cnt = 1
    @inbounds for i = 1:length(c)
        @inbounds for j = 1:length(c[i])
            c[i][j] = cnt
            cnt += 1
        end
    end
    # Second array
    cc = deepcopy(reverse(coloring))
    @inbounds for i = 1:length(cc)
        @inbounds for j = 1:length(cc[i])
            cnt -= 1
            cc[i][j] = cnt
        end
    end
    # Return both
    return c, cc
end




#################################################################
mutable struct ParallelILU0Preconditioner <: AbstractPreconditioner
    A::ExtendableSparseMatrix
    phash::UInt64
    factorization:: _ParallelILU0Preconditioner

    function ParallelILU0Preconditioner()
        p = new()
        p.phash = 0
        p
    end
end


"""
```
ParallelILU0Preconditioner()
ParallelILU0Preconditioner(matrix)
```

Parallel ILU preconditioner with zero fill-in.
"""
function ParallelILU0Preconditioner end

function update!(p::ParallelILU0Preconditioner)
    flush!(p.A)
    Tv=eltype(p.A)
    if p.A.phash!=p.phash
        p.factorization=pilu0(p.A.cscmatrix)
        p.phash=p.A.phash
    else
        pilu0!(p.factorization,p.A.cscmatrix)
    end
    p
end

