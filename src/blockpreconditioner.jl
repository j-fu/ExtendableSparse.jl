mutable struct BlockPreconditioner <: AbstractPreconditioner
    A::ExtendableSparseMatrix
    factorization
    phash::UInt64
    parts::Union{Nothing,Vector{AbstractVector}}
    facts::Vector
    function BlockPreconditioner(;parts=nothing, factorization=LUFactorization) 
        p = new()
        p.phash = 0
        p.parts=parts
        p.factorization=factorization
        p
    end
end


allow_views(::Any)=false

"""
     BlockPreconditioner(parts; factorization=LUFactorization, copywrap=true)
    
Create a block preconditioner from partition of unknowns given by `parts`, a vector of AbstractVectors describing the
indices of the partitions of the matrix. For a matrix of size `n x n`, e.g. parts could be `[ 1:n÷2, (n÷2+1):n]`
or [ 1:2:n, 2:2:n].
Factorization is a callable (Function or struct) which creates a preconditioner (with `ldiv!` methods) from an AbstractMatrix `A`.
"""
function BlockPreconditioner end



function update!(precon::BlockPreconditioner)
    flush!(precon.A)
    nall=sum(length,precon.parts)
    n=size(precon.A,1)
    if nall!=n
	@warn "sum(length,parts)=$(nall) but n=$(n)"
    end

    if isnothing(precon.parts)
        parts=[1:n]
    end
    factorization=precon.factorization
    precon.facts=map(part->factorization(precon.A[part,part]),precon.parts)
end




function LinearAlgebra.ldiv!(p::BlockPreconditioner,v) 
    parts=p.parts
    facts=p.facts
    np=length(parts)

    if allow_views(p.factorization)
        Threads.@threads for ipart=1:np
	    ldiv!(facts[ipart],view(v,parts[ipart]))
        end
    else
        Threads.@threads for ipart=1:np
            vv=v[parts[ipart]]
	    ldiv!(facts[ipart],vv)
            view(v,parts[ipart]).=vv
        end
    end
    v
end

function LinearAlgebra.ldiv!(u,p::BlockPreconditioner,v)
    parts=p.parts
    facts=p.facts
    np=length(parts)
    
    if allow_views(p.factorization)
        Threads.@threads for ipart=1:np
	    ldiv!(view(u,parts[ipart]),facts[ipart],view(v,parts[ipart]))
        end
    else
        Threads.@threads for ipart=1:np
            uu=u[parts[ipart]]
	    ldiv!(uu,facts[ipart],v[parts[ipart]])
            view(u,parts[ipart]).=uu
        end
    end
    u
end

Base.eltype(p::BlockPreconditioner)=eltype(p.facts[1])
