"""
    $(TYPEDEF)    

Sparse matrix where entries are organized as dictionary.
"""
mutable struct SparseMatrixDict{Tv,Ti} <: AbstractSparseMatrixExtension{Tv,Ti}
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

SparseArrays.nnz(m::SparseMatrixDict)=length(m.values)

function SparseArrays.sparse(mat::SparseMatrixDict{Tv,Ti}) where {Tv,Ti} 
    l=length(mat.values)
    I=Vector{Ti}(undef,l)
    J=Vector{Ti}(undef,l)
    V=Vector{Tv}(undef,l)
    i=1
    for (p,v) in mat.values
	I[i]=first(p)
	J[i]=last(p)
	V[i]=v
	i=i+1
    end
    @static if VERSION>=v"1.10"
        return SparseArrays.sparse!(I,J,V,size(mat)...,+)
    else
        return SparseArrays.sparse(I,J,V,size(mat)...,+)
    end
end

function Base.:+(dictmatrix::SparseMatrixDict{Tv,Ti}, cscmatrix::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti} 
    lnew=length(dictmatrix.values)
    if lnew>0
        (;colptr,nzval,rowval,m,n)=cscmatrix
        l=lnew+nnz(cscmatrix)
        I=Vector{Ti}(undef,l)
        J=Vector{Ti}(undef,l)
        V=Vector{Tv}(undef,l)
        i=1
        for icsc=1:length(colptr)-1
            for j=colptr[icsc]:colptr[icsc+1]-1
                I[i]=icsc
                J[i]=rowval[j]
                V[i]=nzval[j]
                i=i+1
            end            
        end
        
        for (p,v) in dictmatrix.values
	    I[i]=first(p)
	    J[i]=last(p)
	    V[i]=v
	    i=i+1
        end
        @static if VERSION>=v"1.10"
            return SparseArrays.sparse!(I,J,V,m,n,+)
        else
            return SparseArrays.sparse(I,J,V,m,n,+)
        end
    end
    cscmatrix
end

function sum!(nodeparts, dictmatrices::Vector{SparseMatrixDict{Tv,Ti}}, cscmatrix::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    lnew=sum(m->length(m.values),dictmatrices)
    if lnew>0
        (;colptr,nzval,rowval,m,n)=cscmatrix
        l=lnew+nnz(cscmatrix)
        I=Vector{Ti}(undef,l)
        J=Vector{Ti}(undef,l)
        V=Vector{Tv}(undef,l)
        i=1
        
        for icsc=1:length(colptr)-1
            for j=colptr[icsc]:colptr[icsc+1]-1
                I[i]=icsc
                J[i]=rowval[j]
                V[i]=nzval[j]
                i=i+1
            end            
        end
        
        ip=1
        for m in dictmatrices
            for (p,v) in m.values
                nodeparts[last(p)]=ip
	        I[i]=first(p)
	        J[i]=last(p)
	        V[i]=v
	        i=i+1
            end
            ip=ip+1
        end
        @static if VERSION>=v"1.10"
            return SparseArrays.sparse!(I,J,V,m,n,+)
        else
            return SparseArrays.sparse(I,J,V,m,n,+)
        end
    end
    return cscmatrix
end
