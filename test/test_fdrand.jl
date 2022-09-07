module test_fdrand
using Test
using SparseArrays
using ExtendableSparse
using LinearAlgebra
using MultiFloats
using ForwardDiff

const Dual64=ForwardDiff.Dual{Float64,Float64,1}
ForwardDiff.value(x::Number)=x

##############################################
function test_fdrand0(T,k,l,m; symmetric=true)
    A=fdrand(T,k,l,m,matrixtype=ExtendableSparseMatrix,symmetric=symmetric)
    jacobi_iteration_matrix=I-inv(Diagonal(A))*A
    ext=extrema(real(eigvals(Float64.(ForwardDiff.value.(Matrix(jacobi_iteration_matrix))))))
    mininv=minimum(inv(Matrix(A)))
    abs(ext[1])<1 &&  abs(ext[2])<1 && mininv>0
end




##############################################
function test_fdrand_coo(T,k,l,m)
    Acsc=fdrand(T,k,l,m,rand=()->1,matrixtype=SparseMatrixCSC)
    Acoo=fdrand(T,k,l,m,rand=()->1,matrixtype=:COO)
    Acsc≈Acoo
end



##############################################
function test_fdrand_update(T, k,l,m)
    A1=fdrand(T, k,l,m,rand=()->1,matrixtype=ExtendableSparseMatrix,update = (A,v,i,j)-> A[i,j]+=v)
    A2=fdrand(T, k,l,m,rand=()->1,matrixtype=ExtendableSparseMatrix,update = (A,v,i,j)-> rawupdateindex!(A,+,v,i,j))
    A3=fdrand(T, k,l,m,rand=()->1,matrixtype=ExtendableSparseMatrix,update = (A,v,i,j)-> updateindex!(A,+,v,i,j))

    A1≈A2 && A1 ≈ A3
end



for T in [Float64, Dual64, Float64x2]
    @test  test_fdrand0(T,100,1,1)
    @test  test_fdrand0(T,10,10,1)
    @test  test_fdrand0(T,5,5,5)
    @test  test_fdrand0(T,100,1,1,symmetric=false)
    @test  test_fdrand0(T,10,10,1,symmetric=false)
    @test  test_fdrand0(T,5,5,5,symmetric=false)
    
    @test  test_fdrand_coo(T,100,1,1)
    @test  test_fdrand_coo(T,10,10,1)
    @test  test_fdrand_coo(T,5,5,5)
    
    @test  test_fdrand_update(T,100,1,1)
    @test  test_fdrand_update(T,10,10,1)
    @test  test_fdrand_update(T,5,5,5)
    
end

end
