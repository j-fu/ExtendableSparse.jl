# Factorizations & Preconditioners 

## Factorizations

In this package, preconditioners and LU factorizations are both seen
as complete or approximate _factorizations_. Correspondingly we provide a common  API for
their creation.


```@autodocs
Modules = [ExtendableSparse]
Pages = ["factorizations.jl"]
Order = [:function, :type]
Private = false
```

## LU Factorizations
```@autodocs
Modules = [ExtendableSparse]
Pages = ["umfpack_lu.jl", "pardiso_lu.jl"]
```

## Preconditioners
```@autodocs
Modules = [ExtendableSparse]
Pages = ["jacobi.jl","ilu0.jl","parallel_jacobi.jl","ilut.jl","amg.jl"]
```

