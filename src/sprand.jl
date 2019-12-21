"""
$(SIGNATURES)

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
    A
end

"""
$(SIGNATURES)

Fill sparse matrix with random entries such that it becomes strictly diagonally
dominant and thus invertible and has a fixed number of nonzeros in
its rows. The matrix bandwidth is bounded by sqrt(n), therefore it 
resembles a typical matrix of a 2D piecewise linear FEM discretization

"""
function sprand_sdd!(A::AbstractSparseMatrix{Tv,Ti},nnzrow::Int; dim=2) where {Tv,Ti}
    m,n=size(A)
    @assert m==n
    nnzrow=min(n,nnzrow)
    bandwidth=convert(Int,ceil(sqrt(n)))
    for i=1:n
        aii=0
        for k=1:nnzrow
            jmin=max(1,i-bandwidth)
            jmax=min(n,i+bandwidth)
            j=rand((jmin:jmax))
            if i!=j
                aij=rand()
                A[i,j]=aij
                aii+=abs(aij)
            end
        end
        A[i,i]=aii+rand() # make it strictly diagonally dominant
    end
    A
end

