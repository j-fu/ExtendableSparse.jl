"""
  $(TYPEDEF)
  Abstract type for a factorization   with ExtandableSparseMatrix. 
  
  Any such preconditioner should have the following fields
````
  A::ExtendableSparseMatrix
  fact
  phash::UInt64
````
and  provide a method
````
  update!(precon)
````   

  The idea is that, depending if the matrix pattern has changed, 
  different steps are needed to update the preconditioner.

  Moreover, they have the ExtendableSparseMatrix `A` as a field and an `update!` method, ensuring 
  consistency after construction.
"""
abstract type AbstractFactorization{Tv, Ti} end

"""
$(TYPEDEF)

Abstract subtype for preconditioners
"""
abstract type AbstractPreconditioner{Tv, Ti} <: AbstractFactorization{Tv, Ti} end

"""
$(TYPEDEF)

Abstract subtype for (full) LU factorizations
"""
abstract type AbstractLUFactorization{Tv, Ti} <: AbstractFactorization{Tv, Ti} end

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
function LinearAlgebra.lu(A::ExtendableSparseMatrix{Tv, Ti}) where {Tv, Ti}
    factorize!(SparspakLU{Tv, Ti}(), A)
end

if USE_GPL_LIBS
    for (Tv) in (:Float64, :ComplexF64)
        @eval begin function LinearAlgebra.lu(A::ExtendableSparseMatrix{$Tv, Ti}) where {Ti}
            factorize!(LUFactorization{$Tv, Ti}(), A)
        end end
    end
end # USE_GPL_LIBS

"""
```
 lufact\rhs
```

Solve  LU factorization problem.
"""
function Base.:\(lufact::AbstractLUFactorization{Tlu, Ti},
                 v::AbstractArray{Tv, 1}) where {Tv, Tlu, Ti}
    ldiv!(similar(v, Tlu), lufact, v)
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
LinearAlgebra.ldiv!(u, fact::AbstractFactorization, v) = ldiv!(u, fact.fact, v)
LinearAlgebra.ldiv!(fact::AbstractFactorization, v) = ldiv!(fact.fact, v)

macro makefrommatrix(fact)
    return quote
        function $(esc(fact))(A::ExtendableSparseMatrix{Tv, Ti}; kwargs...) where {Tv, Ti}
            factorize!($(esc(fact))(; valuetype = Tv, indextype = Ti, kwargs...), A)
        end
        function $(esc(fact))(A::SparseMatrixCSC{Tv, Ti}; kwargs...) where {Tv, Ti}
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

@eval begin
    @makefrommatrix ILU0Preconditioner
    @makefrommatrix ILUZeroPreconditioner
    @makefrommatrix JacobiPreconditioner
    @makefrommatrix ParallelJacobiPreconditioner
    @makefrommatrix ParallelILU0Preconditioner
    @makefrommatrix SparspakLU
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
