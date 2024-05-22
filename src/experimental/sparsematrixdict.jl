mutable struct SparseMatrixDict{Tv,Ti} <: AbstractSparseMatrix{Tv,Ti}
    m::Ti
    n::Ti
    values::Dict{Pair{Ti,Ti}, Tv}
    SparseMatrixDict{Tv,Ti}(m,n) where {Tv,Ti} = new(m,n,Dict{Pair{Ti,Ti}, Tv}())
end

function reset!(m::SparseMatrixDict{Tv,Ti}) where {Tv,Ti}
    m.values=Dict{Pair{Ti,Ti}, Tv}()
end

function Base.setindex!(m::SparseMatrixDict,v,i,j)
	m.values[Pair(i,j)]=v
end

function rawupdateindex!(m::SparseMatrixDict{Tv,Ti},op,v,i,j) where {Tv,Ti}
    p=Pair(i,j)
    m.values[p]=op(get(m.values, p, zero(Tv)),v)
end

function Base.getindex(m::SparseMatrixDict{Tv},i,j) where Tv
    get(m.values,Pair(i,j),zero(Tv)) 
end

Base.size(m::SparseMatrixDict)=(m.m,m.n)

flush!(m::SparseMatrixDict)=nothing

sumlength(mv::Vector{SparseMatrixDict{Tv,Ti}}) where{Tv,Ti}=sum(m->length(m.values),mv)

function SparseArrays.sparse(mv::Vector{SparseMatrixDict{Tv,Ti}}) where {Tv,Ti}
    l=sumlength(mv)
    I=Vector{Ti}(undef,l)
    J=Vector{Ti}(undef,l)
    V=Vector{Tv}(undef,l)
    i=1
    for m in mv
        for (p,v) in m.values
	    I[i]=first(p)
	    J[i]=last(p)
	    V[i]=v
	    i=i+1
        end
    end
    SparseArrays.sparse!(I,J,V,size(mv[1])...,+)
end


SparseArrays.sparse(m::SparseMatrixDict{Tv,Ti}) where {Tv,Ti} = sparse([m])
