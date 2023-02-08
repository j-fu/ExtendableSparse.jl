module test_symmetric
using Test
using ExtendableSparse
using LinearAlgebra

##############################################
function test_symm(n, uplo)
    A = ExtendableSparseMatrix(n, n)
    sprand_sdd!(A)
    b = rand(n)
    flush!(A)
    SA = Symmetric(A, uplo)
    Scsc = Symmetric(A.cscmatrix, uplo)
    
    if ExtendableSparse.USE_GPL_LIBS
        #requires SuiteSparse which is not available on non-GPL builds
        SA \ b ≈ Scsc \ b
    else
        true
    end
end

##############################################
function test_hermitian(n, uplo)
    A = ExtendableSparseMatrix{ComplexF64, Int64}(n, n)
    sprand_sdd!(A)
    flush!(A)
    A.cscmatrix.nzval .= (1.0 + 0.01im) * A.cscmatrix.nzval
    b = rand(n)
    HA = Hermitian(A, uplo)
    Hcsc = Hermitian(A.cscmatrix, uplo)
    if ExtendableSparse.USE_GPL_LIBS
        #requires SuiteSparse which is not available on non-GPL builds
        HA \ b ≈ Hcsc \ b
    else
        true
    end
end

@test test_symm(3, :U)
@test test_symm(3, :L)
@test test_symm(30, :U)
@test test_symm(30, :L)
@test test_symm(300, :U)
@test test_symm(300, :L)

@test test_hermitian(3, :U)
@test test_hermitian(3, :L)
@test test_hermitian(30, :U)
@test test_hermitian(30, :L)
@test test_hermitian(300, :U)
@test test_hermitian(300, :L)

end
