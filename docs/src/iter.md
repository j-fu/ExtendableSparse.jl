# Factorizations & Preconditioners 

## Factorizations

In this package, preconditioners and LU factorizations are subcategories are both seen
as complete or approximate _factorizations_. Correspondingly there is a common API for
their creation.

Factorizations from these package know the matrices which have been factorized.

```@autodocs
Modules = [ExtendableSparse]
Pages = ["factorizations.jl"]
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

