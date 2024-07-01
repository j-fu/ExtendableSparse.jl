mutable struct BlockPreconditioner <: AbstractPreconditioner
    A::ExtendableSparseMatrix
    factorization
    phash::UInt64
    partitioning::Union{Nothing,Vector{AbstractVector}}
    facts::Vector
    function BlockPreconditioner(;partitioning=nothing, factorization=ExtendableSparse.LUFactorization) 
        p = new()
        p.phash = 0
        p.partitioning=partitioning
        p.factorization=factorization
        p
    end
end



"""
     BlockPreconditioner(;partitioning, factorization=LUFactorization)
    
Create a block preconditioner from partition of unknowns given by `partitioning`, a vector of AbstractVectors describing the
indices of the partitions of the matrix. For a matrix of size `n x n`, e.g. partitioning could be `[ 1:n÷2, (n÷2+1):n]`
or [ 1:2:n, 2:2:n].
Factorization is a callable (Function or struct) which allows to create a factorization (with `ldiv!` methods) from a submatrix of A.
"""
function BlockPreconditioner end

"""
    allow_views(::preconditioner_type)

Factorizations on matrix partitions within a block preconditioner may or may not work with array views.
E.g. the umfpack factorization cannot work with views, while ILUZeroPreconditioner can.
Implementing a method for `allow_views` returning `false` resp. `true` allows to dispatch to the proper case.
"""
allow_views(::Any)=false


function update!(precon::BlockPreconditioner)
    flush!(precon.A)
    nall=sum(length,precon.partitioning)
    n=size(precon.A,1)
    if nall!=n
	@warn "sum(length,partitioning)=$(nall) but n=$(n)"
    end

    if isnothing(precon.partitioning)
        partitioning=[1:n]
    end

    np=length(precon.partitioning)
    precon.facts=Vector{Any}(undef,np)
    @tasks for ipart=1:np
        factorization=deepcopy(precon.factorization)
        AP=precon.A[precon.partitioning[ipart],precon.partitioning[ipart]]
        FP=factorization(AP)
        precon.facts[ipart]=FP
    end
end




function LinearAlgebra.ldiv!(p::BlockPreconditioner,v) 
    partitioning=p.partitioning
    facts=p.facts
    np=length(partitioning)

    if allow_views(p.factorization)
        @tasks for ipart=1:np
	    ldiv!(facts[ipart],view(v,partitioning[ipart]))
        end
    else
        @tasks for ipart=1:np
            vv=v[partitioning[ipart]]
	    ldiv!(facts[ipart],vv)
            view(v,partitioning[ipart]).=vv
        end
    end
    v
end

function LinearAlgebra.ldiv!(u,p::BlockPreconditioner,v)
    partitioning=p.partitioning
    facts=p.facts
    np=length(partitioning)
    
    if allow_views(p.factorization)
        @tasks for ipart=1:np
	    ldiv!(view(u,partitioning[ipart]),facts[ipart],view(v,partitioning[ipart]))
        end
    else
        @tasks for ipart=1:np
            uu=u[partitioning[ipart]]
	    ldiv!(uu,facts[ipart],v[partitioning[ipart]])
            view(u,partitioning[ipart]).=uu
        end
    end
    u
end

Base.eltype(p::BlockPreconditioner)=eltype(p.facts[1])
