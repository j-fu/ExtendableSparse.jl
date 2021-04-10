"""
  Abstract type for a factorization
  with ExtandableSparseMatrix. 
  Still beta, so not in the published documentation.
  
  Any such preconditioner should have the following fields
````
  extmatrix
  phash
````
and  methods
````
  update!(precon)
````   
  The idea is that, depending if the matrix pattern has changed, 
  different steps are needed to update the preconditioner.

  Moreover, they have the ExtendableSparseMatrix as a field, ensuring 
  consistency after construction.
"""
abstract type AbstractExtendableSparseFactorization{Tv, Ti} end

abstract type AbstractExtendableSparsePreconditioner{Tv, Ti} <:AbstractExtendableSparseFactorization{Tv, Ti} end

abstract type AbstractExtendableSparseLU{Tv, Ti} <:AbstractExtendableSparseFactorization{Tv, Ti}  end


issolver(::AbstractExtendableSparseLU)=true
issolver(::AbstractExtendableSparsePreconditioner)=false

need_symbolic_update(precon::AbstractExtendableSparseFactorization)=precon.extmatrix.phash!=precon.phash



function options(;kwargs...)
    opt=Dict{Symbol,Any}(
        :kind         => :umfpacklu,
        :ensurelu         => false,
        :droptol         => 1.0e-3,
    )
    for (k,v) in kwargs
        if haskey(opt,Symbol(k))
            opt[Symbol(k)]=v
        end
    end
    opt
end

function factorize(A::ExtendableSparseMatrix; kwargs...)
    opt=options(;kwargs...)
    opt[:kind]==:umfpacklu && return  ExtendableSparseUmfpackLU(A)
    opt[:kind]==:pardiso && return  PardisoLU(A,ps=Pardiso.PardisoSolver())
    opt[:kind]==:mklpardiso && return  PardisoLU(A,ps=Pardiso.MKLPardisoSolver())
    if opt[:ensurelu]
        error("Factorization $(opt[:kind]) is not an lu factorization")
    end
    opt[:kind]==:ilu0 && return ILU0Preconditioner(A)
    opt[:kind]==:ilut && return ILUTPreconditioner(A,droptol=opt[:droptol])
    opt[:kind]==:rsamg && return AMGPreconditioner(A)
    error("Unknown factorization kind: $(opt[:kind])")
end

factorize!(::Nothing, A::ExtendableSparseMatrix; kwargs...) = factorize(A; kwargs...)

Base.:\(lufact::AbstractExtendableSparseLU, v::AbstractArray{T,1} where T)=ldiv!(similar(v), lufact,v)

LinearAlgebra.lu(A::ExtendableSparseMatrix; kwargs...)= factorize(A; ensurelu=true, kwargs...)
LinearAlgebra.lu!(::Nothing, A::ExtendableSparseMatrix; kwargs...)= factorize(A; kwargs...)
LinearAlgebra.lu!(lufact::AbstractExtendableSparseFactorization, A::ExtendableSparseMatrix; kwargs...)=factorize!(lufact,A;kwargs...)

include("jacobi.jl")
include("ilu0.jl")
include("parallel_jacobi.jl")
include("umfpack_lu.jl")
