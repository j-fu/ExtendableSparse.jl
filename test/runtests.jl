using Test
using LinearAlgebra
using SparseArrays
using ExtendableSparse
using Printf

##############################################################
@testset "Constructors" begin
    function test_constructors()
        m1=ExtendableSparseMatrix(10,10)
        m2=ExtendableSparseMatrix(Float16,10,10)
        m3=ExtendableSparseMatrix(Float32,Int16,10,10)
        
        csc=sprand(10,10,0.1)
        m4=ExtendableSparseMatrix(csc)
        sprand!(m4,6)
        flush!(m4)
        true
    end
    @test test_constructors()
end

#################################################################
function test_timing(k,l,m)
    t1=@elapsed fdrand(k,l,m,matrixtype=SparseMatrixCSC)
    t2=@elapsed fdrand(k,l,m,matrixtype=ExtendableSparseMatrix)
    t3=@elapsed fdrand(k,l,m,matrixtype=SparseMatrixLNK)
    @printf("CSC: %.4f EXT: %.4f LNK: %.4f\n",t1,t2,t3)
    t3<t2<t1
end

@testset "Timing" begin
    test_timing(3,3,3)
    @test test_timing(1000,1,1)
    @test test_timing(100,100,1)
    @test test_timing(20,20,20)
end


##################################################################################
function test_assembly(;m=1000,n=1000,xnnz=5000,nsplice=1)
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

@testset "Assembly" begin    
    @test test_assembly(m=10,n=10,xnnz=5)
    @test test_assembly(m=100,n=100,xnnz=500,nsplice=2)
    @test test_assembly(m=1000,n=1000,xnnz=5000,nsplice=3)

    @test test_assembly(m=20,n=10,xnnz=5)
    @test test_assembly(m=200,n=100,xnnz=500,nsplice=2)
    @test test_assembly(m=2000,n=1000,xnnz=5000,nsplice=3)

    @test test_assembly(m=10,n=20,xnnz=5)
    @test test_assembly(m=100,n=200,xnnz=500,nsplice=2)
    @test test_assembly(m=1000,n=2000,xnnz=5000,nsplice=3)

    for irun=1:10
        m=rand((1:10000))
        n=rand((1:10000))
        nnz=rand((1:10000))
        nsplice=rand((1:5))
        @test test_assembly(m=m,n=n,xnnz=nnz,nsplice=nsplice)
    end
end

#####################################################################
function test_transient_construction(;m=10,n=10,d=0.1) where {Tv,Ti}
    csc=sprand(m,n,d)
    lnk=SparseMatrixLNK(csc)
    csc2=SparseMatrixCSC(lnk)
    return csc2==csc
end


@testset "Transient Construction" begin    
    for irun=1:10
        m=rand((1:1000))
        n=rand((1:1000))
        d=0.3*rand()
        @test test_transient_construction(m=m,n=n,d=d)
    end
end

#####################################################################
function test_addition(;m=10,n=10,d=0.1) where {Tv,Ti}
    csc=sprand(m,n,d)
    lnk=SparseMatrixLNK(csc)
    csc2=csc+lnk
    csc2==2*csc
end


function test_operations(n)
    A=ExtendableSparseMatrix(n,n)
    sprand_sdd!(A)
    b=rand(n)
    x=A\b
    Ax=A*x
    b≈ Ax
end


@testset "Operations" begin    
    for irun=1:10
        m=rand((1:1000))
        n=rand((1:1000))
        d=0.3*rand()
        @test test_addition(m=m,n=n,d=d)
    end
        
    @test test_operations(10)
    @test test_operations(100)
    @test test_operations(1000)
end

##############################################
function test_fdrand(k,l,m)
    A=fdrand(k,l,m,matrixtype=ExtendableSparseMatrix)
    ItMat=I-inv(Diagonal(A))*A
    @show ext=extrema(real(eigvals(Matrix(ItMat))))
    @show mininv=minimum(inv(Matrix(A)))
    abs(ext[1])<1 &&  abs(ext[2])<2 && mininv>0
end


@testset "fdrand" begin
  @test  test_fdrand(1000,1,1)
  @test  test_fdrand(20,20,1)
  @test  test_fdrand(10,10,10)
end
    
##############################################

function test_precon(Precon,k,l,m;maxiter=10000)
    A=fdrand(k,l,m,matrixtype=ExtendableSparseMatrix, rand= ()-> 1.0)
    b=ones(size(A,2))
    exact=A\b
    Pl=Precon(A)
    it,hist=simple(A,b,Pl=Pl,maxiter=maxiter,reltol=1.0e-10,log=true)
    r=hist[:resnorm]
    all(x-> x<1,r[end-100:end]./r[end-101:end-1]),norm(it-exact)
end


@testset "preconditioners" begin
    @test   all(isapprox.(test_precon(ILU0Preconditioner,20,20,20),           (true, 1.3535160424212675e-5), rtol=1.0e-5))
    @test   all(isapprox.(test_precon(JacobiPreconditioner,20,20,20),         (true, 2.0406032775945658e-5), rtol=1.0e-5))
    @test   all(isapprox.(test_precon(ParallelJacobiPreconditioner,20,20,20), (true, 2.0406032775945658e-5), rtol=1.0e-5))
end

