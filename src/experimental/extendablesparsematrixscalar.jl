mutable struct ExtendableSparseMatrixScalar{Tm, Tv, Ti <: Integer} <: AbstractExtendableSparseMatrix{Tv, Ti}
    """
    Final matrix data
    """
    cscmatrix::SparseMatrixCSC{Tv, Ti}
    
    """
    Matrix for new entries
    """
    xmatrix::Tm
end


function ExtendableSparseMatrixScalar{Tm, Tv, Ti}(m::Integer,n::Integer) where{Tm, Tv, Ti<:Integer}
    ExtendableSparseMatrixScalar(spzeros(Tv, Ti, m, n),
                                 Tm(m,n)
                                 )
end


function reset!(ext::ExtendableSparseMatrixScalar{Tm,Tv,Ti}) where {Tm,Tv,Ti}
    m,n=size(ext.cscmatrix)
    ext.cscmatrix=spzeros(Tv, Ti, m, n)
    ext.xmatrix=Tm(m,n)
    ext
end


function flush!(ext::ExtendableSparseMatrixScalar{Tm,Tv,Ti}) where{Tm,Tv,Ti}
    ext.cscmatrix=ext.xmatrix+ext.cscmatrix
    ext.xmatrix=Tm(size(ext.cscmatrix)...)
    ext
end
    
function SparseArrays.sparse(ext::ExtendableSparseMatrixScalar)
    flush!(ext)
    ext.cscmatrix
end

function Base.setindex!(ext::ExtendableSparseMatrixScalar,
                        v::Union{Number,AbstractVecOrMat},
                        i::Integer,
                        j::Integer)
    k = findindex(ext.cscmatrix, i, j)
    if k > 0
        ext.cscmatrix.nzval[k] = v
    else
        setindex!(ext.xmatrix,v,i,j)
    end
end


function Base.getindex(ext::ExtendableSparseMatrixScalar,
                       i::Integer,
                       j::Integer)
    k = findindex(ext.cscmatrix, i, j)
    if k > 0
        ext.cscmatrix.nzval[k]
    else
        getindex(ext.xmatrix,i,j)
    end
end

function rawupdateindex!(ext::ExtendableSparseMatrixScalar,
                         op,
                         v,
                         i,
                         j)
    k = findindex(ext.cscmatrix, i, j)
    if k > 0
        ext.cscmatrix.nzval[k] = op(ext.cscmatrix.nzval[k], v)
    else
        rawupdateindex!(ext.xmatrix,op,v,i,j)
    end
end
