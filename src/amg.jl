mutable struct AMGPreconditioner{Tv, Ti} <: AbstractPreconditioner{Tv,Ti}
    A::ExtendableSparseMatrix{Tv,Ti}
    fact
    max_levels::Int
    max_coarse::Int
    function AMGPreconditioner{Tv,Ti}(;max_levels=10, max_coarse=10) where {Tv,Ti}
        precon=new()
        precon.max_levels=max_levels
        precon.max_coarse=max_coarse
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
AMGPreconditioner(;kwargs...)=AMGPreconditioner{Float64,Int64}(;kwargs...)

@eval begin
    @makefrommatrix AMGPreconditioner
end



function update!(precon::AMGPreconditioner{Tv,Ti}) where {Tv,Ti}
    @inbounds flush!(precon.A)
    precon.fact=AlgebraicMultigrid.aspreconditioner(AlgebraicMultigrid.ruge_stuben(precon.A.cscmatrix))
end


