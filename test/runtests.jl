using Test
using LinearAlgebra
using SparseArrays
using ExtendableSparse
using Printf
using BenchmarkTools

using Pardiso
using AlgebraicMultigrid
using IncompleteLU

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

@testset "Updates" begin
    A=ExtendableSparseMatrix(10,10)
    @test nnz(A)==0
    A[1,3]=5
    updateindex!(A,+,6.0,4,5)
    updateindex!(A,+,0.0,2,3)
    @test nnz(A)==2
    rawupdateindex!(A,+,0.0,2,3)
    @test nnz(A)==3
    dropzeros!(A)
    @test nnz(A)==2
    rawupdateindex!(A,+,0.1,2,3)
    @test nnz(A)==3
    dropzeros!(A)
    @test nnz(A)==3
end
#################################################################
function test_timing(k,l,m)
    t1=@belapsed fdrand($k,$l,$m,matrixtype=$SparseMatrixCSC) seconds=0.1
    t2=@belapsed fdrand($k,$l,$m,matrixtype=$ExtendableSparseMatrix)  seconds=0.1
    t3=@belapsed fdrand($k,$l,$m,matrixtype=$SparseMatrixLNK)  seconds=0.1
    @printf("CSC: %.4f EXT: %.4f LNK: %.4f\n",t1*1000,t2*1000,t3*1000 )
    t3<t2<t1
end

@testset "Timing" begin
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
    jacobi_iteration_matrix=I-inv(Diagonal(A))*A
    ext=extrema(real(eigvals(Matrix(jacobi_iteration_matrix))))
    mininv=minimum(inv(Matrix(A)))
    abs(ext[1])<1 &&  abs(ext[2])<1 && mininv>0
end


@testset "fdrand" begin
  @test  test_fdrand(1000,1,1)
  @test  test_fdrand(20,20,1)
  @test  test_fdrand(10,10,10)
end

##############################################
function test_fdrand_coo(k,l,m)
    Acsc=fdrand(k,l,m,rand=()->1,matrixtype=SparseMatrixCSC)
    Acoo=fdrand(k,l,m,rand=()->1,matrixtype=:COO)
    Acsc≈Acoo
end

@testset "fdrand_coo" begin
  @test  test_fdrand_coo(1000,1,1)
  @test  test_fdrand_coo(20,20,1)
  @test  test_fdrand_coo(10,10,10)
end



##############################################
function test_fdrand_update(k,l,m)
    A1=fdrand(k,l,m,rand=()->1,matrixtype=ExtendableSparseMatrix,update = (A,v,i,j)-> A[i,j]+=v)
    A2=fdrand(k,l,m,rand=()->1,matrixtype=ExtendableSparseMatrix,update = (A,v,i,j)-> rawupdateindex!(A,+,v,i,j))
    A3=fdrand(k,l,m,rand=()->1,matrixtype=ExtendableSparseMatrix,update = (A,v,i,j)-> updateindex!(A,+,v,i,j))

    A1≈A2 && A1 ≈ A3
end

@testset "fdrand_update" begin
  @test  test_fdrand_update(1000,1,1)
  @test  test_fdrand_update(20,20,1)
  @test  test_fdrand_update(10,10,10)
end




##############################################

function test_precon(Precon,k,l,m;maxiter=10000)
    A=fdrand(k,l,m,matrixtype=ExtendableSparseMatrix, rand= ()-> 1.0)
    b=ones(size(A,2))
    exact=A\b
    Pl=Precon(A)
    it,hist=simple(A,b,Pl=Pl,maxiter=maxiter,reltol=1.0e-10,log=true)
    r=hist[:resnorm]
    nr=length(r)
    tail=min(100,length(r)÷2)
    all(x-> x<1,r[end-tail:end]./r[end-tail-1:end-1]),norm(it-exact)
end


@testset "preconditioners" begin
    @test   all(isapprox.(test_precon(ILU0Preconditioner,20,20,20),           (true, 1.3535160424212675e-5), rtol=1.0e-5))
    @test   all(isapprox.(test_precon(JacobiPreconditioner,20,20,20),         (true, 2.0406032775945658e-5), rtol=1.0e-5))
    @test   all(isapprox.(test_precon(ParallelJacobiPreconditioner,20,20,20), (true, 2.0406032775945658e-5), rtol=1.0e-5))
    @test   all(isapprox.(test_precon(ILUTPreconditioner,20,20,20),           (true, 1.2719511868322086e-5), rtol=1.0e-5))
    @test   all(isapprox.(test_precon(AMGPreconditioner,20,20,20),            (true, 6.863753664354144e-7), rtol=1.0e-5))
end



##############################################
function test_symmetric(n,uplo)
    A=ExtendableSparseMatrix(n,n)
    sprand_sdd!(A)
    b=rand(n)
    flush!(A)
    SA=Symmetric(A,uplo)
    Scsc=Symmetric(A.cscmatrix,uplo)
    SA\b≈ Scsc\b
end



@testset "symmetric" begin
    @test test_symmetric(3,:U)
    @test test_symmetric(3,:L)
    @test test_symmetric(30,:U)
    @test test_symmetric(30,:L)
    @test test_symmetric(300,:U)
    @test test_symmetric(300,:L)
end



##############################################
function test_hermitian(n,uplo)
    A=ExtendableSparseMatrix{ComplexF64,Int64}(n,n)
    sprand_sdd!(A)
    flush!(A)
    A.cscmatrix.nzval.=(1.0+0.01im)*A.cscmatrix.nzval
    b=rand(n)
    HA=Hermitian(A,uplo)
    Hcsc=Hermitian(A.cscmatrix,uplo)
    HA\b≈ Hcsc\b
end

@testset "hermitian" begin
    @test test_hermitian(3,:U)
    @test test_hermitian(3,:L)
    @test test_hermitian(30,:U)
    @test test_hermitian(30,:L)
    @test test_hermitian(300,:U)
    @test test_hermitian(300,:L)
end



function test_lu(k,l,m; kind=:umfpacklu)
    Acsc=fdrand(k,l,m,rand=()->1,matrixtype=SparseMatrixCSC)
    b=rand(k*l*m)
    LUcsc=lu(Acsc)
    x1csc=LUcsc\b
    for i=1:k*l*m
        Acsc[i,i]+=1.0
    end
    LUcsc=lu!(LUcsc,Acsc)
    x2csc=LUcsc\b

    Aext=fdrand(k,l,m,rand=()->1,matrixtype=ExtendableSparseMatrix)
    LUext=lu(Aext,kind=kind)
    x1ext=LUext\b
    for i=1:k*l*m
        Aext[i,i]+=1.0
    end
    update!(LUext)
    x2ext=LUext\b
    x1csc≈x1ext && x2csc ≈ x2ext
end

function test_lupattern1(k,l,m; kind=:umfpacklu)
    Aext=fdrand(k,l,m,rand=()->1,matrixtype=ExtendableSparseMatrix)
    b=rand(k*l*m)
    LUext=lu(Aext,kind=kind)
    x1ext=LUext\b
    for i=1:k*l*m-3
        Aext[i,i+3]-=1.0e-4
    end
    LUext=lu!(LUext,Aext)
    x2ext=LUext\b
    all(x1ext.< x2ext)
end

function test_lupattern2(k,l,m)
    Aext=fdrand(k,l,m,rand=()->1,matrixtype=ExtendableSparseMatrix)
    b=rand(k*l*m)
    LUext=lu(Aext)
    x1ext=LUext\b
    for i=1:k*l*m-3
        Aext[i,i+3]-=1.0e-4
    end
    update!(LUext)
    x2ext=LUext\b
    all(x1ext.< x2ext)
end

@testset "lu!+update!" begin
    @test test_lu(10,10,10)
    @test test_lu(25,40,1)
    @test test_lu(1000,1,1)

    @test test_lupattern1(10,10,10)
    @test test_lupattern1(25,40,1)
    @test test_lupattern1(1000,1,1)

    @test test_lupattern2(10,10,10)
    @test test_lupattern2(25,40,1)
    @test test_lupattern2(1000,1,1)
end

@testset "pardiso" begin
    @test test_lu(10,10,10,kind=:mklpardiso)
    @test test_lu(25,40,1,kind=:mklpardiso)
    @test test_lu(1000,1,1,kind=:mklpardiso)

    @test test_lupattern1(10,10,10,kind=:mklpardiso)
    @test test_lupattern1(25,40,1,kind=:mklpardiso)
    @test test_lupattern1(1000,1,1,kind=:mklpardiso)
end
