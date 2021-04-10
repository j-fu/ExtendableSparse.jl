"""
$(TYPEDEF)

"""
mutable struct AMGPreconditioner{Tv, Ti} <: AbstractExtendableSparsePreconditioner{Tv,Ti}
    A::ExtendableSparseMatrix{Tv,Ti}
    fact
end

"""
```
AMGPreconditioner(extmatrix)
```
"""
function AMGPreconditioner(A::ExtendableSparseMatrix{Tv,Ti}) where {Tv,Ti}
    @inbounds flush!(A)
    p=AlgebraicMultigrid.aspreconditioner(AlgebraicMultigrid.ruge_stuben(A.cscmatrix))
    AMGPreconditioner{Tv,Ti}(A,p)
end

function update!(precon::AMGPreconditioner{Tv,Ti}) where {Tv,Ti}
    @inbounds flush!(precon.A)
    precon.fact=AlgebraicMultigrid.aspreconditioner(AlgebraicMultigrid.ruge_stuben(precon.A.cscmatrix))
end


