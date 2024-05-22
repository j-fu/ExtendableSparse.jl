mutable struct ExtendableSparseMatrixDict{Tv, Ti <: Integer} <: AbstractExtendableSparseMatrix{Tv, Ti}
    """
    Final matrix data
    """
    cscmatrix::SparseMatrixCSC{Tv, Ti}

    """
        Vector of dictionaries for new entries
    """
    dictmatrix::SparseMatrixDict{Tv,Ti}
end


function ExtendableSparseMatrixDict{Tv, Ti}(n::Integer,m::Integer) where{Tv, Ti<:Integer}
    ExtendableSparseMatrixDict(spzeros(Tv, Ti, m, n),
                               SparseMatrixDict{Tv,Ti}(m,n)
                               )
end

ExtendableSparseMatrixDict(n::Integer,m::Integer)=ExtendableSparseMatrixDict{Float64,Int}(n,m)

function reset!(ext::ExtendableSparseMatrixDict{Tv,Ti}) where {Tv,Ti}
    m,n=size(ext.cscmatrix)
    ext.cscmatrix=spzeros(Tv, Ti, m, n)
    ext.dictmatrix=SparseMatrixDict{Tv,Ti}(m,n)
    ext
end


function flush!(ext::ExtendableSparseMatrixDict{Tv,Ti}) where{Tv,Ti}
    lnew=length(ext.dictmatrix.values)
    if lnew>0
        (;colptr,nzval,rowval,m,n)=ext.cscmatrix
        l=lnew+nnz(ext.cscmatrix)
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

        for (p,v) in ext.dictmatrix.values
	    I[i]=first(p)
	    J[i]=last(p)
	    V[i]=v
	    i=i+1
        end
        
        ext.dictmatrix=SparseMatrixDict{Tv,Ti}(m,n)
        ext.cscmatrix=SparseArrays.sparse!(I,J,V,m,n,+)
    end
    ext
end
    
function SparseArrays.sparse(ext::ExtendableSparseMatrixDict)
    flush!(ext)
    ext.cscmatrix
end

function Base.setindex!(ext::ExtendableSparseMatrixDict{Tv, Ti},
                        v::Union{Number,AbstractVecOrMat},
                        i::Integer,
                        j::Integer) where {Tv, Ti}
    k = findindex(ext.cscmatrix, i, j)
    if k > 0
        ext.cscmatrix.nzval[k] = v
    else
        setindex!(ext.dictmatrix,v,i,j)
    end
end


function Base.getindex(ext::ExtendableSparseMatrixDict{Tv, Ti},
                       i::Integer,
                       j::Integer) where {Tv, Ti <: Integer}
    k = findindex(ext.cscmatrix, i, j)
    if k > 0
        ext.cscmatrix.nzval[k]
    else
        getindex(ext.dictmatrix,i,j)
    end
end

function rawupdateindex!(ext::ExtendableSparseMatrixDict{Tv, Ti},
                         op,
                         v,
                         i,
                         j) where {Tv, Ti <: Integer}
    k = findindex(ext.cscmatrix, i, j)
    if k > 0
        ext.cscmatrix.nzval[k] = op(ext.cscmatrix.nzval[k], v)
    else
        rawupdateindex!(ext.dictmatrix,op,v,i,j)
    end
end
