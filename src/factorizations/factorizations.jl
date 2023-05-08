"""
  $(TYPEDEF)

Abstract type for a factorization   with ExtandableSparseMatrix. 

This type is meant to be a "type flexible" (with respect to the matrix element type)
and lazily construcdet (can be constructed without knowing the matrix, and updated later)
LU factorization or preconditioner. It wraps different concrete, type fixed factorizations
which shall provide the usual `ldiv!` methods.

Any such preconditioner/factorization `MyFact` should have the following fields
````
  A::ExtendableSparseMatrix
  factorization
  phash::UInt64
````
and  provide methods
````
  MyFact(;kwargs...) 
  update!(precon::MyFact)
````   

The idea is that, depending if the matrix pattern has changed, 
different steps are needed to update the preconditioner.

    
"""
abstract type AbstractFactorization end

"""
$(TYPEDEF)

Abstract subtype for preconditioners
"""
abstract type AbstractPreconditioner <: AbstractFactorization end

"""
$(TYPEDEF)

Abstract subtype for (full) LU factorizations
"""
abstract type AbstractLUFactorization <: AbstractFactorization end

"""
```
issolver(factorization)
```

Determine if factorization is a solver or not
"""
issolver(::AbstractLUFactorization) = true
issolver(::AbstractPreconditioner) = false

"""
```
factorize!(factorization, matrix)
```

Update or create factorization, possibly reusing information from the current state.
This method is aware of pattern changes.
"""
function factorize!(p::AbstractFactorization, A::ExtendableSparseMatrix)
    p.A = A
    update!(p)
    p
end


factorize!(p::AbstractFactorization, A::SparseMatrixCSC)=factorize!(p,ExtendableSparseMatrix(A))
"""
```
lu!(factorization, matrix)
```

Update LU factorization, possibly reusing information from the current state.
This method is aware of pattern changes.

If `nothing` is passed as first parameter, [`factorize`](@ref) is called.
"""
function LinearAlgebra.lu!(lufact::AbstractFactorization, A::ExtendableSparseMatrix)
    factorize!(lufact, A)
end

"""
```
lu(matrix)
```

Create LU factorization. It calls the LU factorization form Sparspak.jl, unless GPL components
are allowed  in the Julia sysimage and the floating point type of the matrix is Float64 or Complex64.
In that case, Julias standard `lu` is called, which is realized via UMFPACK.
"""
function lu  end

if USE_GPL_LIBS
    function LinearAlgebra.lu(A::ExtendableSparseMatrix)
        factorize!(LUFactorization(), A)
    end
else
    function LinearAlgebra.lu(A::ExtendableSparseMatrix)
        factorize!(SparspakLU(), A)
    end
end # USE_GPL_LIBS

"""
```
 lufact\rhs
```

Solve  LU factorization problem.
"""
function Base.:\(lufact::AbstractLUFactorization,v)
    ldiv!(similar(v), lufact, v)
end

"""
```
update!(factorization)
```
Update factorization after matrix update.
"""
update!(::AbstractFactorization)

"""
```
ldiv!(u,factorization,v)
ldiv!(factorization,v)
```

Solve factorization.
"""
LinearAlgebra.ldiv!(u, fact::AbstractFactorization, v) = ldiv!(u, fact.factorization, v)
LinearAlgebra.ldiv!(fact::AbstractFactorization, v) = ldiv!(fact.factorization, v)



""""
    @makefrommatrix(fact)

For an AbstractFactorization `MyFact`, provide methods
```
    MyFact(A::ExtendableSparseMatrix; kwargs...)
    MyFact(A::SparseMatrixCSC; kwargs...)
```
"""
macro makefrommatrix(fact)
    return quote
        function $(esc(fact))(A::ExtendableSparseMatrix; kwargs...)
            factorize!($(esc(fact))(;kwargs...), A)
        end
        function $(esc(fact))(A::SparseMatrixCSC; kwargs...)
            $(esc(fact))(ExtendableSparseMatrix(A); kwargs...)
        end
    end
end

include("jacobi.jl")
include("ilu0.jl")
include("iluzero.jl")
include("parallel_jacobi.jl")
include("parallel_ilu0.jl")
include("sparspak.jl")
include("blockpreconditioner.jl")

@eval begin
    @makefrommatrix ILU0Preconditioner
    @makefrommatrix ILUZeroPreconditioner
    @makefrommatrix PointBlockILUZeroPreconditioner
    @makefrommatrix JacobiPreconditioner
    @makefrommatrix ParallelJacobiPreconditioner
    @makefrommatrix ParallelILU0Preconditioner
    @makefrommatrix SparspakLU
    @makefrommatrix UpdateteableBlockpreconditioner
    @makefrommatrix BlockPreconditioner
end

if USE_GPL_LIBS
    #requires SuiteSparse which is not available in non-GPL builds
    include("umfpack_lu.jl")
    include("cholmod_cholesky.jl")

    @eval begin
        @makefrommatrix LUFactorization
        @makefrommatrix CholeskyFactorization
    end
else
    const LUFactorization = SparspakLU
end
