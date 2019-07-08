"""
$(TYPEDSIGNATURES)

Fill empty sparse matrix A with random nonzero elements from interval [1,2]
using incremental assembly.

"""
function sprand!(A::AbstractSparseMatrix{Tv,Ti},xnnz::Int) where {Tv,Ti}
    m,n=size(A)
    for i=1:xnnz
        i=rand((1:m))
        j=rand((1:n))
        a=1.0+rand(Tv)
        A[i,j]+=a
    end
end

