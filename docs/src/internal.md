# Internal API

## Linked List Sparse Matrix format

```@autodocs
Modules = [ExtendableSparse]
Pages = ["sparsematrixlnk.jl"]
```

## Some methods for SparseMatrixCSC

```@autodocs
Modules = [ExtendableSparse]
Pages = ["sparsematrixcsc.jl"]
```
## New API 
Under development - aimed at multithreading
```@autodocs
Modules = [ExtendableSparse]
Pages = ["abstractsparsematrixextension.jl",
    "abstractextendablesparsematrixcsc.jl",
    "sparsematrixdilnkc.jl",
    "genericextendablesparsematrixcsc.jl",
    "genericmtextendablesparsematrixcsc.jl"]
```


## Misc methods

```@docs
ExtendableSparse.@makefrommatrix :: Tuple{Any}
```
