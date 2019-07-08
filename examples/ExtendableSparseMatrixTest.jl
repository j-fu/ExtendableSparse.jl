module ExtendableSparseTest

using ExtendableSparse
using SparseArrays
using Printf

import PyPlot

function randmatx!(A::AbstractSparseMatrix{Tv,Ti}, m::Ti, n::Ti,xnnz::Ti) where {Tv,Ti}
    for inz=1:xnnz
        i=rand((1:m))
        j=rand((1:n))
        a=1.0+rand(Float64)
        A[i,j]+=a
    end

end

function testmatx!(A::AbstractSparseMatrix{Tv,Ti}, m::Ti, n::Ti, xnnz::Ti) where {Tv,Ti}
    S=spzeros(m,n)
    for inz=1:xnnz
        i=rand((1:m))
        j=rand((1:n))
        a=1.0+rand(Float64)
        @printf("%d %d %e\n",i,j,a)
        S[i,j]+=a
        A[i,j]+=a
        @printf("nnz(S)=%d nnz(A)=%d\n",nnz(S),nnz(A))
        @assert(nnz(S)==nnz(A))
    end
    @printf("nnz(S)=%d nnz(A)=%d\n",nnz(S),nnz(A))
    @assert(nnz(S)==nnz(A))
    (I,J,V)=findnz(S)
    for inz=1:nnz(S)
        @assert(A[I[inz],J[inz]]==V[inz])
    end
end

function check(;m=10,n=10,nnz=3)
    extmat1=SparseMatrixExtension{Float64,Int64}(m,n)
    testmatx!(extmat1,m,n,nnz)
end


function benchmark(;n=100,m=100,nnz=500, pyplot=false)
    
    mat=spzeros(Float64,Int64,n,n)
    @time randmatx!(mat,m,n,nnz)

    extmat=SparseMatrixExtension{Float64,Int64}(n,n)
    @time randmatx!(extmat,m,n,nnz)
    
    
    xextmat=ExtendableSparseMatrixCSC(n,n,spzeros(Float64,Int64,n,n),SparseMatrixExtension{Float64,Int64}(n,n))
    @time randmatx!(xextmat,m,n,nnz)



    
    if pyplot
        PyPlot.clf()
        PyPlot.spy(extmat,markersize=1)
        PyPlot.show()
    end
    # for i=1:n
    #     println(rand(1:n))
    # end
end

end
