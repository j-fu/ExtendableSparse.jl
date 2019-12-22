module ExtendableSparseTest
using ExtendableSparse
using SparseArrays
using Printf



function test_assembly(;m=1000,n=1000,xnnz=5000,nsplice=1)
    println("test_assembly( m=$(m), n=$(n), xnnz=$(xnnz), nsplice=$(nsplice))")
    A=ExtendableSparseMatrix{Float64,Int64}(m,n)
    m,n=size(A)
    S=spzeros(m,n)
    for isplice=1:nsplice
        for inz=1:xnnz
            i=rand((1:m))
            j=rand((1:n))
            a=1.0+rand(Float64)
            S[i,j]+=a
            A[i,j]+=a
        end
        flush!(A)
        for j=1:n
            @assert(issorted(A.cscmatrix.rowval[A.cscmatrix.colptr[j]:A.cscmatrix.colptr[j+1]-1]))
        end
        @assert(nnz(S)==nnz(A))

        (I,J,V)=findnz(S)
        for inz=1:nnz(S)
            @assert(A[I[inz],J[inz]]==V[inz])
        end

        (I,J,V)=findnz(A)
        for inz=1:nnz(A)
            @assert(S[I[inz],J[inz]]==V[inz])
        end

    end
    return true
end


function test_transient_construction(;m=10,n=10,d=0.1) where {Tv,Ti}
    println("test_transient_construction( m=$(m), n=$(n), d=$(d))")
    csc=sprand(m,n,d)
    lnk=SparseMatrixLNK(csc)
    csc2=SparseMatrixCSC(lnk)
    return csc2==csc
end

function test_addition(;m=10,n=10,d=0.1) where {Tv,Ti}
    println("test_addition( m=$(m), n=$(n), d=$(d))")
    csc=sprand(m,n,d)
    lnk=SparseMatrixLNK(csc)
    csc2=csc+lnk
    csc2==2*csc
end



function test_timing(;n=10000,m=10000,nnz=50000)
    println("test_timing( m=$(m), n=$(n), nnz=$(nnz))")
    
    println("SparseMatrixCSC:")
    mat=spzeros(Float64,Int64,m,n)
    @time sprand!(mat,nnz)
    
    println("SparseMatrixLNK:")
    extmat=SparseMatrixLNK{Float64,Int64}(m,n)
    @time sprand!(extmat,nnz)
    
    println("ExtendableSparseMatrix:")
    xextmat=ExtendableSparseMatrix{Float64,Int64}(m,n)
    @time begin
        sprand!(xextmat,nnz)
        @inbounds flush!(xextmat)
    end
    true
end

function test_constructors()
    println("test_constructors()")
    m1=ExtendableSparseMatrix(10,10)
    m2=ExtendableSparseMatrix(Float16,10,10)
    m3=ExtendableSparseMatrix(Float32,Int16,10,10)

    csc=sprand(10,10,0.1)
    m4=ExtendableSparseMatrix(csc)
    sprand!(m4,6)
    flush!(m4)
    
    true
end


function test_operations(n)
    println("test_operations(n=$(n))")
    A=ExtendableSparseMatrix(n,n)
    sprand_sdd!(A)
    b=rand(n)
    @time x=A\b
    Ax=A*x
    b≈ Ax
end
end
