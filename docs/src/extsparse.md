# Sparse matrix handling

## Matrix creation and update API

```@autodocs
Modules = [ExtendableSparse]
Pages = ["extendable.jl"]
```

```@docs
ExtendableSparse.lu
LinearAlgebra.lu!
LinearAlgebra.ldiv!
```

## Handling of homogeneous Dirichlet BC
```@docs
mark_dirichlet
eliminate_dirichlet!
eliminate_dirichlet
```

## Test matrix creation

```@autodocs
Modules = [ExtendableSparse]
Pages = ["sprand.jl"]
```
