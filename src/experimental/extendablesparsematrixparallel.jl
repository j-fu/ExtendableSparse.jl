mutable struct ExtendableSparseMatrixXParallel{Tm, Tv, Ti <: Integer} <: AbstractExtendableSparseMatrix{Tv, Ti}
    """
    Final matrix data
    """
    cscmatrix::SparseMatrixCSC{Tv, Ti}

    """
        Vector of dictionaries for new entries
    """
    xmatrices::Vector{Tm}

    nodeparts::Vector{Ti}
    partnodes::Vector{Vector{Ti}}
    colparts::Vector{Vector{Ti}}
end


function ExtendableSparseMatrixXParallel{Tm, Tv, Ti}(n,m,p::Integer) where{Tm, Tv, Ti}
    ExtendableSparseMatrixXParallel(spzeros(Tv, Ti, m, n),
                                    [Tm(m,n) for i=1:p],
                                    zeros(Ti,n),
                                    Vector{Ti}[],
                                    Vector{Ti}[]
                                    )
end

function partcolors!(ext::ExtendableSparseMatrixXParallel{Tm,Tv,Ti}, partcolors) where {Tm, Tv, Ti}
    ncol=maximum(partcolors)
    colparts=[Ti[] for i=1:ncol]
    for i=1:length(partcolors)
        push!(colparts[partcolors[i]],i)
    end
    ext.colparts=colparts
    ext
end

function ExtendableSparseMatrixXParallel{Tm, Tv, Ti}(n,m, pc::Vector) where{Tm, Tv, Ti}
    ext=ExtendableSparseMatrixXParallel(m,n,length(pc))
    partcolors!(ext,pc)
end


function reset!(ext::ExtendableSparseMatrixXParallel{Tm,Tv,Ti},p::Integer) where {Tm,Tv,Ti}
    m,n=size(ext.cscmatrix)
    ext.cscmatrix=spzeros(Tv, Ti, m, n)
    ext.xmatrices=[Tm(m,n) for i=1:p]
    ext.nodeparts.=zero(Ti)
    ext
end

function reset!(ext::ExtendableSparseMatrixXParallel)
    reset!(ext,length(ext.xmatrices))
end

function reset!(ext::ExtendableSparseMatrixXParallel,pc::Vector)
    reset!(ext,length(pc))
    partcolors!(ext,pc)
end

function flush!(ext::ExtendableSparseMatrixXParallel{Tm,Tv,Ti}) where{Tm,Tv,Ti}
    ext.cscmatrix=sum!(ext.nodeparts, ext.xmatrices, ext.cscmatrix)
    np=length(ext.xmatrices)
    (m,n)=size(ext.cscmatrix)
    ext.xmatrices=[Tm(m,n) for i=1:np]
 
    npts::Vector{Ti}=ext.nodeparts
    pn=zeros(Ti,np)
    for i=1:n
        npi=npts[i]
        if npi>0
            pn[npi]+=1
        end
    end
    partnodes=[zeros(Int,pn[i]) for i=1:np]
    pn.=1
    for i=1:n
        npi=ext.nodeparts[i]
        if npi>0
            partnodes[npi][pn[npi]]=i
            pn[npi]+=1
        end
    end
    ext.partnodes=partnodes
    ext
end


function SparseArrays.sparse(ext::ExtendableSparseMatrixXParallel)
    flush!(ext)
    ext.cscmatrix
end



function Base.setindex!(ext::ExtendableSparseMatrixXParallel,
                        v::Union{Number,AbstractVecOrMat},
                        i::Integer,
                        j::Integer)
    k = findindex(ext.cscmatrix, i, j)
    if k > 0
        ext.cscmatrix.nzval[k] = v
    else
        error("use rawupdateindex! for new entries into ExtendableSparseMatrixXParallel")
    end
end


function Base.getindex(ext::ExtendableSparseMatrixXParallel,
                       i::Integer,
                       j::Integer)
    k = findindex(ext.cscmatrix, i, j)
    if k > 0
        return ext.cscmatrix.nzval[k]
    elseif sum(nnz,ext.xmatrices) == 0
        return zero(eltype(ext.cscmatrix))
    else
        error("flush! ExtendableSparseMatrixXParallel before using getindex")
    end
end

function rawupdateindex!(ext::ExtendableSparseMatrixXParallel,
                         op,
                         v,
                         i,
                         j,
                         tid)
    k = findindex(ext.cscmatrix, i, j)
    if k > 0
        ext.cscmatrix.nzval[k] = op(ext.cscmatrix.nzval[k], v)
    else
        rawupdateindex!(ext.xmatrices[tid],op,v,i,j)
    end
end

function LinearAlgebra.mul!(r, ext::ExtendableSparseMatrixXParallel, x)
    A=ext.cscmatrix
    colparts=ext.colparts
    partnodes=ext.partnodes
    rows = SparseArrays.rowvals(A)
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
