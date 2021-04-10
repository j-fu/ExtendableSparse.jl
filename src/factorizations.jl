"""
  $(TYPEDEF)
  Abstract type for a factorization   with ExtandableSparseMatrix. 
  
  Any such preconditioner should have the following fields
````
  A
  precondata
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



#
# Print default dict for interpolation into docstrings
#
function _myprint(dict::Dict{Symbol,Any})
    lines_out=IOBuffer()
    for (k,v) in dict
        println(lines_out,"  - $(k): $(v)")
    end
    String(take!(lines_out))
end


"""
```
const default_options
```

Default options for various factorizations:

$(_myprint(default_options))

"""
const default_options=Dict{Symbol,Any}(
    :kind         => :umfpacklu,
    :droptol      => 1.0e-3,
    :ensurelu     => false,
)



"""
```
options(;kwargs...)
```
Set default options and blend them with `kwargs`
"""
function options(;kwargs...)
    opt=copy(default_options)
    for (k,v) in kwargs
        if haskey(opt,Symbol(k))
            opt[Symbol(k)]=v
        end
    end
    opt
end


"""
```
factorize(matrix)
factorize(matrix; kind=:umfpacklu)
```
Default Julia LU factorization based on UMFPACK.

```
factorize(matrix; kind=:pardiso)
factorize(matrix; kind=:mklpardiso)
```
LU factorization based on pardiso. For using this, you need to issue
`using Pardiso`. `:pardiso` uses the  solver from [pardiso-project.org](https://pardiso-project.org),
while `:mklpardiso` uses the early 2000's fork in Intel's MKL library.

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
function factorize(A::ExtendableSparseMatrix; kwargs...)
    opt=options(;kwargs...)
    opt[:kind]==:umfpacklu && return  ExtendableSparseUmfpackLU(A)
    opt[:kind]==:pardiso && return  PardisoLU(A,ps=Pardiso.PardisoSolver())
    opt[:kind]==:mklpardiso && return  PardisoLU(A,ps=Pardiso.MKLPardisoSolver())
    if opt[:ensurelu]
        error("Factorization $(opt[:kind]) is not an lu factorization")
    end
    opt[:kind]==:ilu0 && return ILU0Preconditioner(A)
    opt[:kind]==:jacobi && return JacobiPreconditioner(A)
    opt[:kind]==:pjacobi && return ParallelJacobiPreconditioner(A)
    opt[:kind]==:ilut && return ILUTPreconditioner(A,droptol=opt[:droptol])
    opt[:kind]==:rsamg && return AMGPreconditioner(A)
    error("Unknown factorization kind: $(opt[:kind])")
end

"""
```
factorize!(factorization_or_nothing, matrix; kwargs...)
```

Update factorization, possibly reusing information from the current state.
This method is aware of pattern changes.

If `nothing` is passed as first parameter, [`factorize`](@ref) is called.
"""
factorize!(::Nothing, A::ExtendableSparseMatrix; kwargs...) = factorize(A; kwargs...)


"""
```
lu(matrix)
lu(matrix,kind=:pardiso)
lu(matrix,kind=:mklpardiso)
```
Wrapper for [`factorize`](@ref) restricted to lu factorizations.
"""
LinearAlgebra.lu(A::ExtendableSparseMatrix; kwargs...)= factorize(A; ensurelu=true, kwargs...)


"""
```
lu!(factorization_or_nothing, matrix; kwargs...)
```

Update LU factorization, possibly reusing information from the current state.
This method is aware of pattern changes.

If `nothing` is passed as first parameter, [`factorize`](@ref) is called.
"""
LinearAlgebra.lu!(::Nothing, A::ExtendableSparseMatrix; kwargs...)= factorize(A; kwargs...)
LinearAlgebra.lu!(lufact::AbstractExtendableSparseFactorization, A::ExtendableSparseMatrix; kwargs...)=factorize!(lufact,A;kwargs...)

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
LinearAlgebra.ldiv!




include("jacobi.jl")
include("ilu0.jl")
include("parallel_jacobi.jl")
include("umfpack_lu.jl")
