
mutable struct SparspakLU <: AbstractLUFactorization
    A::Union{ExtendableSparseMatrix, Nothing}
    factorization::Union{Sparspak.SparseSolver.SparseSolver, Nothing}
    phash::UInt64
end

function SparspakLU()
    fact = SparspakLU(nothing, nothing, 0)
end

"""
```
SparspakLU() 
SparspakLU(matrix) 
```

LU factorization based on [Sparspak.jl](https://github.com/PetrKryslUCSD/Sparspak.jl) (P.Krysl's Julia re-implementation of Sparspak by George & Liu)
"""
function SparspakLU end

function update!(lufact::SparspakLU)
    flush!(lufact.A)
    if lufact.A.phash != lufact.phash
        lufact.factorization = sparspaklu(lufact.A.cscmatrix)
    else
        sparspaklu!(lufact.factorization, lufact.A.cscmatrix)
    end
end
