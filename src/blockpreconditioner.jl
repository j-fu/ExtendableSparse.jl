struct CopyWrapper
    factorization
end

CopyWrapper(A::AbstractMatrix,factorization)= CopyWrapper(factorization(A))

LinearAlgebra.ldiv!(cw::CopyWrapper,v)=ldiv!(cw.factorization,collect(v))

function LinearAlgebra.ldiv!(u,cw::CopyWrapper,v)
    vv=collect(v)
    uu=collect(u)
    ldiv!(uu,cw.factorization,vv)
    u.=uu
end

mutable struct BlockPreconditioner{Tv, Ti} <: AbstractPreconditioner{Tv, Ti}
    A::ExtendableSparseMatrix{Tv, Ti}
    factorization
    phash::UInt64
    parts::Union{Nothing,Vector{AbstractVector}}
    facts::Vector
    function BlockPreconditioner{Tv, Ti}(;parts=nothing, factorization=LUFactorization) where {Tv, Ti}
        p = new()
        p.phash = 0
        p.parts=parts
        p.factorization=factorization
        p
    end
end


needs_copywrap(::Any)=true

"""
    BlockPreconditioner(parts; valuetype=Float64, indextype=Int64, factorization=LUFactorization, copywrap=true)

Create a block preconditioner from partition of unknowns given by parts.
Factorization is a callable (Funtion or struct) which creates a preconditioner (with `ldiv!` methods) from an AbstractMatrix A.
"""
function BlockPreconditioner(;parts=nothing, valuetype::Type = Float64, indextype::Type = Int64, factorization= LUFactorization)
    BlockPreconditioner{valuetype, indextype}(;parts,factorization)
end

function update!(precon::BlockPreconditioner{Tv, Ti}) where {Tv, Ti}
    flush!(precon.A)
    nall=sum(length,precon.parts)
    n=size(precon.A,1)
    if nall!=n
	@warn "sum(length,parts)=$(nall) but n=$(n)"
    end

    if isnothing(precon.parts)
        parts=[1:n]
    end
    @show needs_copywrap(precon.factorization)
    if needs_copywrap(precon.factorization)
        factorization= A-> CopyWrapper(A,precon.factorization)
    else
        factorization=precon.factorization
    end
    precon.facts=map(part->factorization(precon.A[part,part]),precon.parts)
end




function LinearAlgebra.ldiv!(p::BlockPreconditioner,v) 
    (;parts,facts)=p
    np=length(parts)
    Threads.@threads for ipart=1:np
	ldiv!(facts[ipart],view(v,parts[ipart]))
    end
    v
end

function LinearAlgebra.ldiv!(u,p::BlockPreconditioner,v)
    (;parts,facts)=p
    np=length(parts)
    Threads.@threads for ipart=1:np
	ldiv!(view(u,parts[ipart]), facts[ipart], view(v,parts[ipart]))
    end
    u
end

Base.eltype(p::BlockPreconditioner)=eltype(p.facts[1])
