mutable struct ExtendableSparseMatrixParallelDict{Tv, Ti <: Integer} <: AbstractExtendableSparseMatrix{Tv, Ti}
    """
    Final matrix data
    """
    cscmatrix::SparseMatrixCSC{Tv, Ti}

    """
        Linked list structure holding data of extension
    """
    dictmatrices::Vector{SparseMatrixDict{Tv,Ti}}

    nodeparts::Vector{Ti}
    partnodes::Vector{Vector{Ti}}
    colparts::Vector{Vector{Ti}}
end


function ExtendableSparseMatrixParallelDict{Tv, Ti}(n,m,p::Integer) where{Tv, Ti}
    ExtendableSparseMatrixParallelDict(spzeros(Tv, Ti, m, n),
                                       [SparseMatrixDict{Tv,Ti}(m,n) for i=1:p],
                                       zeros(Ti,n),
                                       Vector{Ti}[],
                                       Vector{Ti}[]
                                       )
end

function partcolors!(ext::ExtendableSparseMatrixParallelDict{Tv,Ti}, partcolors) where {Tv, Ti}
    ncol=maximum(partcolors)
    colparts=[Ti[] for i=1:ncol]
    for i=1:length(partcolors)
        push!(colparts[partcolors[i]],i)
    end
    ext.colparts=colparts
    ext
end

function ExtendableSparseMatrixParallelDict{Tv, Ti}(n,m,pc::Vector) where{Tv, Ti}
    ext=ExtendableSparseMatrixParallelDict(m,n,length(pc))
    partcolors!(ext,pc)
end


ExtendableSparseMatrixParallelDict(n,m,p)=ExtendableSparseMatrixParallelDict{Float64,Int}(n,m,p)


function reset!(ext::ExtendableSparseMatrixParallelDict{Tv,Ti},p::Integer) where {Tv,Ti}
    m,n=size(ext.cscmatrix)
    ext.cscmatrix=spzeros(Tv, Ti, m, n)
    ext.dictmatrices=[SparseMatrixDict{Tv,Ti}(m,n) for i=1:p]
    ext.nodeparts.=zero(Ti)
    ext
end

function reset!(ext::ExtendableSparseMatrixParallelDict{Tv,Ti}) where {Tv,Ti}
    reset!(ext,length(ext.dictmatrices))
end

function reset!(ext::ExtendableSparseMatrixParallelDict{Tv,Ti},pc::Vector) where {Tv,Ti}
    reset!(ext,length(pc))
    partcolors!(ext,pc)
end


function flush!(ext::ExtendableSparseMatrixParallelDict{Tv,Ti}) where{Tv,Ti}
    lnew=sumlength(ext.dictmatrices)
    if lnew>0
        (;colptr,nzval,rowval,m,n)=ext.cscmatrix
        l=lnew+nnz(ext.cscmatrix)
        I=Vector{Ti}(undef,l)
        J=Vector{Ti}(undef,l)
        V=Vector{Tv}(undef,l)
        i=1
        ip=1
        for m in ext.dictmatrices
            for (p,v) in m.values
                ext.nodeparts[first(p)]=ip
	        I[i]=first(p)
	        J[i]=last(p)
	        V[i]=v
	        i=i+1
            end
            ip=ip+1
        end
        
        for icsc=1:length(colptr)-1
            for j=colptr[icsc]:colptr[icsc+1]-1
                I[i]=icsc
                J[i]=rowval[j]
                V[i]=nzval[j]
                i=i+1
            end            
        end

        np=length(ext.dictmatrices)
        ext.dictmatrices=[SparseMatrixDict{Tv,Ti}(m,n) for i=1:np]
        ext.cscmatrix=SparseArrays.sparse!(I,J,V,m,n,+)
        
        n,m=size(ext)
        pn=zeros(Int,np)
        for i=1:n
            if ext.nodeparts[i]>0
                pn[ext.nodeparts[i]]+=1
            end
        end
        partnodes=[zeros(Int,pn[i]) for i=1:np]
        pn.=1
        for i=1:n
            if ext.nodeparts[i]>0
                ip=ext.nodeparts[i]
                partnodes[ip][pn[ip]]=i
                pn[ip]+=1
            end
        end
        ext.partnodes=partnodes
    end
    ext
end

function SparseArrays.sparse(ext::ExtendableSparseMatrixParallelDict)
    flush!(ext)
    ext.cscmatrix
end



function Base.setindex!(ext::ExtendableSparseMatrixParallelDict{Tv, Ti},
                        v::Union{Number,AbstractVecOrMat},
                        i::Integer,
                        j::Integer) where {Tv, Ti}
    k = findindex(ext.cscmatrix, i, j)
    if k > 0
        ext.cscmatrix.nzval[k] = v
    else
        error("use rawupdateindex! for new entries into ExtendableSparseMatrixParallelDict")
    end
end


function Base.getindex(ext::ExtendableSparseMatrixParallelDict{Tv, Ti},
                       i::Integer,
                       j::Integer) where {Tv, Ti <: Integer}
    k = findindex(ext.cscmatrix, i, j)
    if k > 0
        return ext.cscmatrix.nzval[k]
    elseif sumlength(ext.dictmatrices) == 0
        return zero(Tv)
    else
        error("flush! ExtendableSparseMatrixParallelDict before using getindex")
    end
end

function rawupdateindex!(ext::ExtendableSparseMatrixParallelDict{Tv, Ti},
                         op,
                         v,
                         i,
                         j,
                         tid) where {Tv, Ti <: Integer}
    k = findindex(ext.cscmatrix, i, j)
    if k > 0
        ext.cscmatrix.nzval[k] = op(ext.cscmatrix.nzval[k], v)
    else
        rawupdateindex!(ext.dictmatrices[tid],op,v,i,j)
    end
end

function LinearAlgebra.mul!(r, ext::ExtendableSparseMatrixParallelDict{Tv,Ti}, x) where {Tv,Ti}
    A=ext.cscmatrix
    colparts=ext.colparts
    partnodes=ext.partnodes
    rows = rowvals(A)
    vals = nonzeros(A)

    r.=zero(Tv)
    m,n=size(A)
    for icol=1:length(colparts)
        part=colparts[icol]
        @tasks for ip=1:length(part)
            @inbounds begin
                for j in partnodes[part[ip]]
                    for i in nzrange(A,j)
                        row = rows[i]
                        val = vals[i]
                        r[row]+=val*x[j]
                    end
                end
            end
        end
    end
    r
end

