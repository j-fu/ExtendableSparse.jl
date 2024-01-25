"""
$(SIGNATURES)

Return index corresponding to entry [i,j] in the array of nonzeros,
if the entry exists, otherwise, return 0.
"""
function findindex(csc::SparseMatrixCSC{T}, i, j) where {T}
    if !(1 <= i <= csc.m && 1 <= j <= csc.n)
        throw(BoundsError())
    end
    r1 = Int(csc.colptr[j])
    r2 = Int(csc.colptr[j + 1] - 1)
    if r1 > r2
        return 0
    end

    # See sparsematrix.jl
    r1 = searchsortedfirst(csc.rowval, i, r1, r2, Base.Forward)
    if (r1 > r2 || csc.rowval[r1] != i)
        return 0
    end
    return r1
end

"""
$(SIGNATURES)

Update element of the matrix  with operation `op` whithout
introducing new nonzero elements.

This can replace the following code and save one index
search during acces:

```@example
using ExtendableSparse # hide
using SparseArrays # hide
A=spzeros(3,3)
A[1,2]+=0.1
A
```

```@example
using ExtendableSparse # hide
using SparseArrays # hide
A=spzeros(3,3)
updateindex!(A,+,0.1,1,2)
A
```

"""
function updateindex!(csc::SparseMatrixCSC{Tv, Ti}, op, v, i, j) where {Tv, Ti <: Integer}
    k = findindex(csc, i, j)
    if k > 0 # update existing value
        csc.nzval[k] = op(csc.nzval[k], v)
    else # insert new value
        csc[i, j] = op(zero(Tv), v)
    end
    csc
end

"""
$(SIGNATURES)

Trival flush! method for allowing to run the code with both `ExtendableSparseMatrix` and
`SparseMatrixCSC`.
"""
flush!(csc::SparseMatrixCSC) = csc

"""
$(SIGNATURES)

Hash of csc matrix pattern. 
"""
phash(csc::SparseMatrixCSC) = hash((hash(csc.colptr), hash(csc.rowval)))
# probably no good idea to add two hashes, so we hash them together.

"""
    pattern_equal(a::SparseMatrixCSC,b::SparseMatrixCSC)

Check if sparsity patterns of two SparseMatrixCSC objects are equal.
This is generally faster than comparing hashes.
"""
function pattern_equal(a::SparseMatrixCSC, b::SparseMatrixCSC)
    a.colptr == b.colptr && a.rowval == b.rowval
end


function pointblock(A::SparseMatrixCSC,blocksize)
    SparseMatrixCSC(pointblock(ExtendableSparseMatrix(A),blocksize))
end

"""
   mark_dirichlet(A; penalty=1.0e20)

Return boolean vector marking Dirichlet nodes, known by `A[i,i]>=penalty`
"""
function mark_dirichlet(A::SparseMatrixCSC; penalty=1.0e20)
    colptr=A.colptr
    rowval=A.rowval
    nzval=A.nzval
    n=A.n
    dirichlet=zeros(Bool,n)
    for i=1:n
        for j=colptr[i]:colptr[i+1]-1
            if rowval[j]==i && nzval[j]>=penalty
                dirichlet[i]=true
            end
        end
    end
    dirichlet
end

"""
    eliminate_dirichlet!(A,dirichlet_marker)

Eliminate dirichlet nodes in matrix by setting 
```julia
    A[:,i]=0; A[i,:]=0; A[i,i]=1
```
for a node `i` marked as Dirichlet.

Returns A.
"""
function eliminate_dirichlet!(A::SparseMatrixCSC,dirichlet)
    colptr=A.colptr
    rowval=A.rowval
    nzval=A.nzval
    n=A.n
    for i=1:n
        # set off-diagonal column indiced to zero
        if !iszero(dirichlet[i])
            for j=colptr[i]:colptr[i+1]-1
                if rowval[j]==i
                    nzval[j]=1
                else
                    nzval[j]=0
                end
            end
        end
        # set off-diagonal row indices to zero
        for j=colptr[i]:colptr[i+1]-1
            if rowval[j]!=i && !iszero(dirichlet[rowval[j]])
                nzval[j]=0
            end
        end
    end
    A
end

"""
    eliminate_dirichlet(A,dirichlet_marker)

Create a copy B of A sharing the sparsity pattern.
Eliminate dirichlet nodes in B by setting 
```julia
    B[:,i]=0; B[i,:]=0; B[i,i]=1
```
for a node `i` marked as Dirichlet.

Returns B.
"""
function eliminate_dirichlet(A::SparseMatrixCSC,dirichlet)
    B=SparseMatrixCSC(A.m,A.n,A.colptr,A.rowval,copy(A.nzval))
    eliminate_dirichlet!(B,dirichlet)
end
