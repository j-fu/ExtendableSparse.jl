module ExtendableSparseAlgebraicMultigridExt

using ExtendableSparse

isdefined(Base, :get_extension) ? using AlgebraicMultigrid : using ..AlgebraicMultigrid

import ExtendableSparse: @makefrommatrix, AbstractPreconditioner, update!

mutable struct AMGPreconditioner <: AbstractPreconditioner
    A::ExtendableSparseMatrix
    factorization::AlgebraicMultigrid.Preconditioner
    max_levels::Int
    max_coarse::Int
    function ExtendableSparse.AMGPreconditioner(; max_levels = 10, max_coarse = 10)
        precon = new()
        precon.max_levels = max_levels
        precon.max_coarse = max_coarse
        precon
    end
end


@eval begin
    @makefrommatrix  ExtendableSparse.AMGPreconditioner
end

function update!(precon::AMGPreconditioner)
    @inbounds flush!(precon.A)
    precon.factorization = AlgebraicMultigrid.aspreconditioner(AlgebraicMultigrid.ruge_stuben(precon.A.cscmatrix))
end

allow_views(::AMGPreconditioner)=true
allow_views(::Type{AMGPreconditioner})=true

end
