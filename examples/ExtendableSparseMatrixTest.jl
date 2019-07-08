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
    flush!(A)
    @printf("nnz(S)=%d nnz(A)=%d\n",nnz(S),nnz(A))
    @assert(nnz(S)==nnz(A))
    (I,J,V)=findnz(S)
    for inz=1:nnz(S)
        @assert(A[I[inz],J[inz]]==V[inz])
    end
end

function check(;m=10,n=10,nnz=3)
    extmat1=SparseMatrixExtension(Float64,Int64,m,n)
    testmatx!(extmat1,m,n,nnz)

    extmat2=ExtendableSparseMatrix(Float64,Int64,m,n)
    testmatx!(extmat2,m,n,nnz)
end


function benchmark(;n=100,m=100,nnz=500, pyplot=false)
    
    mat=spzeros(Float64,Int64,m,n)
    @time randmatx!(mat,m,n,nnz)

    extmat=SparseMatrixExtension(Float64,Int64,m,n)
    @time randmatx!(extmat,m,n,nnz)
    
    
    xextmat=ExtendableSparseMatrix(Float64,Int64,m,n)
    @time begin
        randmatx!(xextmat,m,n,nnz)
        @inbounds flush!(xextmat)
    end


    
    if pyplot
        PyPlot.clf()
        PyPlot.spy(extmat,markersize=1)
        PyPlot.show()
    end
    # for i=1:n
    #     println(rand(1:n))
    # end
end

function tsplice(;m=10, n=10, nnz=20)
    xnnz=nnz
    
    A=ExtendableSparseMatrix(Float64,Int64,m,n)
    for inz=1:xnnz
        i=rand((1:m))
        j=rand((1:n))
        a=1.0+rand(Float64)
        A[i,j]+=a
    end
    flush!(A)
    for inz=1:2
        i=rand((1:m))
        j=rand((1:n))
        a=1.0+rand(Float64)
        A[i,j]+=a
    end
    flush!(A)
end

end
