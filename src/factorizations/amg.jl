mutable struct AMGPreconditioner <: AbstractPreconditioner
    A::ExtendableSparseMatrix
    factorization::AlgebraicMultigrid.Preconditioner
    max_levels::Int
    max_coarse::Int
    function AMGPreconditioner(; max_levels = 10, max_coarse = 10)
        precon = new()
        precon.max_levels = max_levels
        precon.max_coarse = max_coarse
        precon
    end
end

"""
```
AMGPreconditioner(;max_levels=10, max_coarse=10)
AMGPreconditioner(matrix;max_levels=10, max_coarse=10)
```

Create the  [`AMGPreconditioner`](@ref) wrapping the Ruge-Stüben AMG preconditioner from [AlgebraicMultigrid.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl)
"""
function AMGPreconditioner end

@eval begin
    @makefrommatrix AMGPreconditioner
end

function update!(precon::AMGPreconditioner)
    @inbounds flush!(precon.A)
    precon.factorization = AlgebraicMultigrid.aspreconditioner(AlgebraicMultigrid.ruge_stuben(precon.A.cscmatrix))
end

allow_views(::AMGPreconditioner)=true
allow_views(::Type{AMGPreconditioner})=true
