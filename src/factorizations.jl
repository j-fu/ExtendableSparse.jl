"""
  $(TYPEDEF)
  Abstract type for a factorization   with ExtandableSparseMatrix. 
  
  Any such preconditioner should have the following fields
````
  A
  fact
  phash
````
and  methods
````
  factorize(A; kwargs)
  ldiv!(u, precon,v)
  ldiv!(precon,v)
  issolver(precon)
  update!(precon)
  factorize!(precon,A; kwargs)
````   
  The idea is that, depending if the matrix pattern has changed, 
  different steps are needed to update the preconditioner.

  Moreover, they have the ExtendableSparseMatrix as a field, ensuring 
  consistency after construction.
"""
abstract type AbstractExtendableSparseFactorization{Tv, Ti} end

"""
$(TYPEDEF)
Abstract subtype for preconditioners
"""
abstract type AbstractExtendableSparsePreconditioner{Tv, Ti} <:AbstractExtendableSparseFactorization{Tv, Ti} end

"""
$(TYPEDEF)
Abstract subtype for (full) LU factorizations
"""
abstract type AbstractExtendableSparseLU{Tv, Ti} <:AbstractExtendableSparseFactorization{Tv, Ti}  end



"""
```
issolver(factorization)
```

Determine if factorization is a solver or not
"""
issolver(::AbstractExtendableSparseLU)=true
issolver(::AbstractExtendableSparsePreconditioner)=false


"""
```
factorize(matrix; kind=:ilu0)
```
Create the [`ILU0Preconditioner`](@ref) from this package.

```
factorize(matrix; kind=:jacobi)
```
Create the [`JacobiPreconditioner`](@ref) from this package.

```
factorize(matrix; kind=:pjacobi)
```
Create the [`ParallelJacobiPreconditioner`](@ref) from this package.


```
factorize(matrix; kind=:ilut, droptol=1.0e-3)
```
Create the [`ILUTPreconditioner`](@ref) wrapping the one 
from [IncompleteLU.jl](https://github.com/haampie/IncompleteLU.jl)
For using this, you need to issue `using IncompleteLU`


```
factorize(matrix; kind=:rsamg)
```
Create the  [`AMGPreconditioner`](@ref) wrapping the Ruge-Stüben AMG preconditioner from [AlgebraicMultigrid.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl)
"""


#=
Create factoriztion object without matrix,
and only have factorize! as API.

LUFactorization
CholeskyFactorization
PardisoLUFactorization
ILU0Preconditioner
JacobiPreconditioner
ILUTPreconditioner
AMGPreconditioner(rugestueben/smagg)
=#


"""
```
factorize!(factorization, matrix)
```

Update or create factorization, possibly reusing information from the current state.
This method is aware of pattern changes.
"""
function factorize!(p::AbstractExtendableSparseFactorization, A::ExtendableSparseMatrix)
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
LinearAlgebra.lu!(lufact::AbstractExtendableSparseFactorization, A::ExtendableSparseMatrix)=factorize!(lufact,A)

"""
```
 lufact\rhs
```

Solve  LU factorization problem.
"""
Base.:\(lufact::AbstractExtendableSparseLU, v::AbstractArray{T,1} where T)=ldiv!(similar(v), lufact,v)


"""
```
update!(factorization)
```
Update factorization after matrix update.
"""
update!(::AbstractExtendableSparseFactorization)


"""
```
ldiv!(u,factorization,v)
ldiv!(factorization,v)
```

Solve factorization.
"""
LinearAlgebra.ldiv!(u,fact::AbstractExtendableSparseFactorization, v)=ldiv!(u, fact.fact, v)
LinearAlgebra.ldiv!(fact::AbstractExtendableSparseFactorization, v)=ldiv!(fact.fact,v)




include("jacobi.jl")
include("ilu0.jl")
include("parallel_jacobi.jl")
include("umfpack_lu.jl")
include("cholmod_cholesky.jl")
