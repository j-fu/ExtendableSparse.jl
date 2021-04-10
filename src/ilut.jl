"""
$(TYPEDEF)

ILU(T) preconditioner
"""
mutable struct ILUTPreconditioner{Tv, Ti} <: AbstractExtendableSparsePreconditioner{Tv,Ti}
    A::ExtendableSparseMatrix
    fact::IncompleteLU.ILUFactorization{Tv,Ti}
    droptol::Float64
end

"""
```
ILUTPreconditioner(matrix; droptol=1.0e-3)
```
"""
function ILUTPreconditioner(A::ExtendableSparseMatrix; droptol=1.0e-3)
    @inbounds flush!(A)
    ILUTPreconditioner(A,IncompleteLU.ilu(A.cscmatrix,τ=droptol),droptol)
end

function update!(precon::ILUTPreconditioner, A::ExtendableSparseMatrix)
    A=precon.A
    @inbounds flush!(A)
    precon.fact=IncompleteLU.ilu(A.cscmatrix,τ=precon.droptol)
end

