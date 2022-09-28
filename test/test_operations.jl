module test_operations
using Test
using SparseArrays
using ExtendableSparse


#####################################################################
function test_addition(;m=10,n=10,d=0.1) where {Tv,Ti}
    csc=sprand(m,n,d)
    lnk=SparseMatrixLNK(csc)
    csc2=csc+lnk
    csc2==2*csc
end

function test_invert(n)
    A=ExtendableSparseMatrix(n,n)
    sprand_sdd!(A)
    b=rand(n)
    x=A\b
    Ax=A*x
    b≈ Ax
end


for irun=1:10
    m=rand((1:1000))
    n=rand((1:1000))
    d=0.3*rand()
    @test test_addition(m=m,n=n,d=d)
end

@test test_invert(10)
@test test_invert(100)
@test test_invert(1000)

end
