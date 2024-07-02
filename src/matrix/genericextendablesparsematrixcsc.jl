mutable struct GenericExtendableSparseMatrixCSC{Tm<:AbstractSparseMatrixExtension, Tv, Ti <: Integer} <: AbstractExtendableSparseMatrixCSC{Tv, Ti}
    """
    Final matrix data
    """
    cscmatrix::SparseMatrixCSC{Tv, Ti}
    
    """
    Matrix for new entries
    """
    xmatrix::Tm
end


function GenericExtendableSparseMatrixCSC{Tm, Tv, Ti}(m::Integer,n::Integer) where{Tm<:AbstractSparseMatrixExtension, Tv, Ti<:Integer}
    GenericExtendableSparseMatrixCSC(spzeros(Tv, Ti, m, n),
                                 Tm(m,n)
                                 )
end


nnznew(ext::GenericExtendableSparseMatrixCSC)=nnz(ext.xmatrix)

function reset!(ext::GenericExtendableSparseMatrixCSC{Tm,Tv,Ti}) where {Tm,Tv,Ti}
    m,n=size(ext.cscmatrix)
    ext.cscmatrix=spzeros(Tv, Ti, m, n)
    ext.xmatrix=Tm(m,n)
    ext
end


function flush!(ext::GenericExtendableSparseMatrixCSC{Tm,Tv,Ti}) where{Tm,Tv,Ti}
    if nnz(ext.xmatrix)>0
        ext.cscmatrix=ext.xmatrix+ext.cscmatrix
        ext.xmatrix=Tm(size(ext.cscmatrix)...)
    end
    ext
end
    
function SparseArrays.sparse(ext::GenericExtendableSparseMatrixCSC)
    flush!(ext)
    ext.cscmatrix
end

function Base.setindex!(ext::GenericExtendableSparseMatrixCSC,
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


function Base.getindex(ext::GenericExtendableSparseMatrixCSC,
                       i::Integer,
                       j::Integer)
    k = findindex(ext.cscmatrix, i, j)
    if k > 0
        ext.cscmatrix.nzval[k]
    else
        getindex(ext.xmatrix,i,j)
    end
end

function rawupdateindex!(ext::GenericExtendableSparseMatrixCSC,
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

function updateindex!(ext::GenericExtendableSparseMatrixCSC,
                         op,
                         v,
                         i,
                         j)
    k = findindex(ext.cscmatrix, i, j)
    if k > 0
        ext.cscmatrix.nzval[k] = op(ext.cscmatrix.nzval[k], v)
    else
        updateindex!(ext.xmatrix,op,v,i,j)
    end
end

