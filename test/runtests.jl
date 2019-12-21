using Test
include("ExtendableSparseTest.jl")


@test ExtendableSparseTest.constructors()
@test ExtendableSparseTest.benchmark()


@time begin

    
    @test ExtendableSparseTest.check(m=10,n=10,nnz=5)
    @test ExtendableSparseTest.check(m=100,n=100,nnz=500,nsplice=2)
    @test ExtendableSparseTest.check(m=1000,n=1000,nnz=5000,nsplice=3)

    @test ExtendableSparseTest.check(m=20,n=10,nnz=5)
    @test ExtendableSparseTest.check(m=200,n=100,nnz=500,nsplice=2)
    @test ExtendableSparseTest.check(m=2000,n=1000,nnz=5000,nsplice=3)

    @test ExtendableSparseTest.check(m=10,n=20,nnz=5)
    @test ExtendableSparseTest.check(m=100,n=200,nnz=500,nsplice=2)
    @test ExtendableSparseTest.check(m=1000,n=2000,nnz=5000,nsplice=3)

    for irun=1:10
        m=rand((1:10000))
        n=rand((1:10000))
        nnz=rand((1:10000))
        nsplice=rand((1:5))
        @test ExtendableSparseTest.check(m=m,n=n,nnz=nnz,nsplice=nsplice)
    end
        
    
end
