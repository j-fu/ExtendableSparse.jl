using Test
include("ExtendableSparseTest.jl")


@test ExtendableSparseTest.test_constructors()
@test ExtendableSparseTest.test_timing()


@time begin

    
    @test ExtendableSparseTest.test_assembly(m=10,n=10,xnnz=5)
    @test ExtendableSparseTest.test_assembly(m=100,n=100,xnnz=500,nsplice=2)
    @test ExtendableSparseTest.test_assembly(m=1000,n=1000,xnnz=5000,nsplice=3)

    @test ExtendableSparseTest.test_assembly(m=20,n=10,xnnz=5)
    @test ExtendableSparseTest.test_assembly(m=200,n=100,xnnz=500,nsplice=2)
    @test ExtendableSparseTest.test_assembly(m=2000,n=1000,xnnz=5000,nsplice=3)

    @test ExtendableSparseTest.test_assembly(m=10,n=20,xnnz=5)
    @test ExtendableSparseTest.test_assembly(m=100,n=200,xnnz=500,nsplice=2)
    @test ExtendableSparseTest.test_assembly(m=1000,n=2000,xnnz=5000,nsplice=3)

    for irun=1:10
        m=rand((1:10000))
        n=rand((1:10000))
        nnz=rand((1:10000))
        nsplice=rand((1:5))
        @test ExtendableSparseTest.test_assembly(m=m,n=n,xnnz=nnz,nsplice=nsplice)
    end
        
    for irun=1:10
        m=rand((1:1000))
        n=rand((1:1000))
        d=0.3*rand()
        @test ExtendableSparseTest.test_transient_construction(m=m,n=n,d=d)
    end
        
    for irun=1:10
        m=rand((1:1000))
        n=rand((1:1000))
        d=0.3*rand()
        @test ExtendableSparseTest.test_addition(m=m,n=n,d=d)
    end
        
    @test ExtendableSparseTest.test_operations(10)
    @test ExtendableSparseTest.test_operations(100)
    @test ExtendableSparseTest.test_operations(1000)
end
