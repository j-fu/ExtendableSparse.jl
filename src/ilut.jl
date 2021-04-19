mutable struct ILUTPreconditioner{Tv, Ti} <: AbstractPreconditioner{Tv,Ti}
    A::ExtendableSparseMatrix
    fact::IncompleteLU.ILUFactorization{Tv,Ti}
    droptol::Float64
    function ILUTPreconditioner{Tv,Ti}(;droptol=1.0e-3) where {Tv,Ti}
        p=new()
        p.droptol=droptol
        p
    end
end

"""
```
ILUTPreconditioner(;droptol=1.0e-3)
ILUTPreconditioner(matrix; droptol=1.0e-3)
```

Create the [`ILUTPreconditioner`](@ref) wrapping the one 
from [IncompleteLU.jl](https://github.com/haampie/IncompleteLU.jl)
For using this, you need to issue `using IncompleteLU`.
"""
ILUTPreconditioner(;kwargs...)=ILUTPreconditioner{Float64,Int64}(;kwargs...)

@eval begin
    @makefrommatrix ILUTPreconditioner
end

function update!(precon::ILUTPreconditioner)
    A=precon.A
    @inbounds flush!(A)
    precon.fact=IncompleteLU.ilu(A.cscmatrix,τ=precon.droptol)
end

