module ExtendableSparseIncompleteLUExt

using ExtendableSparse

isdefined(Base, :get_extension) ? using IncompleteLU : using ..IncompleteLU

import ExtendableSparse: @makefrommatrix, AbstractPreconditioner, update!

mutable struct ILUTPreconditioner <: AbstractPreconditioner
    A::ExtendableSparseMatrix
    factorization::IncompleteLU.ILUFactorization
    droptol::Float64
    function ExtendableSparse.ILUTPreconditioner(; droptol = 1.0e-3)
        p = new()
        p.droptol = droptol
        p
    end
end


@eval begin
    @makefrommatrix ExtendableSparse.ILUTPreconditioner
end

function update!(precon::ILUTPreconditioner)
    A = precon.A
    @inbounds flush!(A)
    precon.factorization = IncompleteLU.ilu(A.cscmatrix; τ = precon.droptol)
end
end

