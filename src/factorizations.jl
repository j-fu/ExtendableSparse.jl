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

  Moreover, they have the ExtendableSparseMatrix as a field, ensuring 
  consistency after construction.
"""
abstract type AbstractFactorization{Tv, Ti} end

"""
$(TYPEDEF)

Abstract subtype for preconditioners
"""
abstract type AbstractPreconditioner{Tv, Ti} <:AbstractFactorization{Tv, Ti} end

"""
$(TYPEDEF)

Abstract subtype for (full) LU factorizations
"""
abstract type AbstractLUFactorization{Tv, Ti} <:AbstractFactorization{Tv, Ti}  end



"""
```
issolver(factorization)
```

Determine if factorization is a solver or not
"""
issolver(::AbstractLUFactorization)=true
issolver(::AbstractPreconditioner)=false



"""
```
factorize!(factorization, matrix)
```

Update or create factorization, possibly reusing information from the current state.
This method is aware of pattern changes.
"""
function factorize!(p::AbstractFactorization, A::ExtendableSparseMatrix)
    p.A=A
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
LinearAlgebra.lu!(lufact::AbstractFactorization, A::ExtendableSparseMatrix)=factorize!(lufact,A)

LinearAlgebra.lu(A::ExtendableSparseMatrix)=factorize!(LUFactorization(),A)


"""
```
 lufact\rhs
```

Solve  LU factorization problem.
"""
Base.:\(lufact::AbstractLUFactorization, v::AbstractArray{T,1} where T)=ldiv!(similar(v), lufact,v)


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
LinearAlgebra.ldiv!(u,fact::AbstractFactorization, v)=ldiv!(u, fact.fact, v)
LinearAlgebra.ldiv!(fact::AbstractFactorization, v)=ldiv!(fact.fact,v)



macro makefrommatrix(fact)
    return quote
        $(esc(fact))(A::ExtendableSparseMatrix{Tv,Ti}; kwargs...) where {Tv,Ti} = factorize!($(esc(fact))(;valuetype=Tv, indextype=Ti,  kwargs...),A)
        $(esc(fact))(A::SparseMatrixCSC{Tv,Ti}; kwargs...) where {Tv,Ti} = $(esc(fact))(ExtendableSparseMatrix(A); kwargs...)
    end
end

include("jacobi.jl")
include("ilu0.jl")
include("parallel_jacobi.jl")
include("parallel_ilu0.jl")

@eval begin

    @makefrommatrix ILU0Preconditioner
    @makefrommatrix JacobiPreconditioner
    @makefrommatrix ParallelJacobiPreconditioner
    @makefrommatrix ParallelILU0Preconditioner

end

if Base.USE_GPL_LIBS
#requires SuiteSparse which is not available in non-GPL builds
    include("umfpack_lu.jl")
    include("cholmod_cholesky.jl")
    
    @eval begin
    	
        @makefrommatrix LUFactorization
        @makefrommatrix CholeskyFactorization
    
    end
end




