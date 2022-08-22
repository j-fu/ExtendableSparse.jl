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

Fill sparse matrix  with random entries such that  it becomes strictly
diagonally  dominant  and  thus  invertible and  has  a  fixed  number
`nnzrow` (default: 4) of nonzeros in its rows. The matrix bandwidth is
bounded by  sqrt(n) in order to resemble  a typical matrix of  a 2D
piecewise linear FEM discretization.
"""
function sprand_sdd!(A::AbstractSparseMatrix{Tv,Ti}; nnzrow=4) where {Tv,Ti}
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



"""
$(SIGNATURES)

After setting  all nonzero entries  to zero, fill resp.  update matrix
with finite  difference discretization data  on a unit  hypercube. See
[`fdrand`](@ref) for documentation of the parameters.

It is required that `size(A)==(N,N)` where `N=nx*ny*nz`
"""
function fdrand!(A::T,
                 nx,ny=1,nz=1;
                 update= (A,v,i,j)-> A[i,j]+=v,
                 rand=()-> rand()) where T<:AbstractMatrix
    
    sz=size(A)
    N=nx*ny*nz
    if sz[1]!=N || sz[2]!=N
        error("Matrix size mismatch")
    end
    
    _flush!(m::ExtendableSparseMatrix)=flush!(m)
    _flush!(m::SparseMatrixCSC)=m
    _flush!(m::SparseMatrixLNK)=m
    _flush!(m::AbstractMatrix)=m

    _nonzeros(m::Matrix)=vec(m)
    _nonzeros(m::ExtendableSparseMatrix)=nonzeros(m)
    _nonzeros(m::SparseMatrixLNK)=m.nzval
    _nonzeros(m::SparseMatrixCSC)=nonzeros(m)

    zero!(A::AbstractMatrix{T}) where T = A.=zero(T)
    zero!(A::SparseMatrixCSC{T,Ti}) where {T,Ti} = _nonzeros(A).=zero(T)
    zero!(A::ExtendableSparseMatrix{T,Ti}) where {T,Ti} = _nonzeros(A).=zero(T)
    zero!(A::SparseMatrixLNK{T,Ti}) where {T,Ti} = _nonzeros(A).=zero(T)

    zero!(A)
    
    function update_pair(A,v,i,j)
        update(A,-v,i,j)
        update(A,-v,j,i)
        update(A,v,i,i)
        update(A,v,j,j)
    end
    

    
    hx=1.0/nx
    hy=1.0/ny
    hz=1.0/nz

    nxy=nx*ny
    l=1
    for k=1:nz
        for j=1:ny
            for i=1:nx
                if i<nx
                    update_pair(A,rand()*hy*hz/hx,l,l+1)
                end
                if i==1|| i==nx
                    update(A,rand()*hy*hz,l,l)
                end
                if j<ny
                    update_pair(A,rand()*hx*hz/hy,l,l+nx)
                end
                if ny>2&&(j==1|| j==ny)
                    update(A,rand()*hx*hz,l,l)
                end
                if k<nz
                    update_pair(A,rand()*hx*hy/hz,l,l+nxy)
                end
                if nz>2&&(k==1|| k==nz)
                    update(A,rand()*hx*hy,l,l)
                end
                l=l+1
            end
        end
    end
    _flush!(A)
end

"""
$(SIGNATURES)

Create SparseMatrixCSC via COO intermedite arrrays
"""

function fdrand_coo(nx,ny=1,nz=1;
                    rand=()-> rand())
    N=nx*ny*nz
    I=zeros(Int64,0)
    J=zeros(Int64,0)
    V=zeros(Float64,0)
    
    function update(v,i,j)
        push!(I,i)
        push!(J,j)
        push!(V,v)
    end

    function update_pair(v,i,j)
        update(-v,i,j)
        update(-v,j,i)
        update(v,i,i)
        update(v,j,j)
    end
    
    hx=1.0/nx
    hy=1.0/ny
    hz=1.0/nz

    nxy=nx*ny
    l=1
    for k=1:nz
        for j=1:ny
            for i=1:nx
                if i<nx
                    update_pair(rand()*hy*hz/hx,l,l+1)
                end
                if i==1|| i==nx
                    update(rand()*hy*hz,l,l)
                end
                if j<ny
                    update_pair(rand()*hx*hz/hy,l,l+nx)
                end
                if ny>2&&(j==1|| j==ny)
                    update(rand()*hx*hz,l,l)
                end
                if k<nz
                    update_pair(rand()*hx*hy/hz,l,l+nxy)
                end
                if nz>2&&(k==1|| k==nz)
                    update(rand()*hx*hy,l,l)
                end
                l=l+1
            end
        end
    end
    sparse(I,J,V)
end
"""
$(SIGNATURES)

Create matrix  for a mock  finite difference operator for  a diffusion
problem with random coefficients on a unit hypercube ``\\Omega\\subset\\mathbb R^d``.
with ``d=1`` if  `nx>1 && ny==1 && nz==1`, ``d=2`` if  `nx>1 && ny>1 && nz==1` and
``d=3`` if  `nx>1 && ny>1 && nz>1` .

```math
    \\begin{align*}
             -\\nabla a \\nabla u &= f&&  \\text{in}\\;  \\Omega  \\\\
    a\\nabla u\\cdot \\vec n + bu &=g && \\text{on}\\;  \\partial\\Omega
    \\end{align*}
```

The matrix is irreducibly diagonally dominant, has positive main diagonal entries 
and nonpositive off-diagonal entries, hence it has the M-Property.
Therefore, its inverse will be a dense matrix with positive entries. 
Moreover it is symmmetric, and  positive definite, and the spectral radius
of the  Jacobi iteration matrix ``\rho(I-D(A)^{-1}A)<1`` .

Parameters+ default values:

|   Parameter + default vale        | Description                        |
| ---------------------------------:|:---------------------------------- |
|                              `nx` | Number of unknowns in x direction  |
|                              `ny` | Number of unknowns in y direction  |
|                              `nz` | Number of unknowns in z direction  |   
|    `matrixtype = SparseMatrixCSC` | Matrix type                        |
|  `update = (A,v,i,j)-> A[i,j]+=v` | Element update function            |
|    `rand =()-> rand()`            | Random number generator            |
                                                                             
The sparsity structure is fixed to an orthogonal grid, resulting in a 3, 5 or 7
diagonal matrix depending on dimension. The entries
are random unless e.g.  `rand=()->1` is passed as random number generator.
Tested for Matrix, SparseMatrixCSC,  ExtendableSparseMatrix, SparseMatrixLNK and `:COO`

"""
function fdrand(nx,ny=1,nz=1;
                matrixtype::Union{Type,Symbol}=SparseMatrixCSC,
                update = (A,v,i,j)-> A[i,j]+=v,
                rand= ()-> 0.1+rand(),symmetric=true)
    N=nx*ny*nz
    if matrixtype==:COO
        A=fdrand_coo(nx,ny,nz, rand=rand)
        else
        if matrixtype==ExtendableSparseMatrix
            A=ExtendableSparseMatrix(N,N)
        elseif matrixtype==SparseMatrixLNK
            A=SparseMatrixLNK(N,N)
        elseif matrixtype==SparseMatrixCSC
            A=spzeros(N,N)
        elseif matrixtype==Tridiagonal
            A=Tridiagonal(zeros(nx-1),zeros(nx),zeros(nx-1))
        elseif matrixtype==Matrix
            A=zeros(N,N)
        end
        A=fdrand!(A,nx,ny,nz,update = update, rand=rand)
    end
    if symmetric
        A
    else
        Diagonal([rand() for i=1:size(A,1)])*A
    end
end



