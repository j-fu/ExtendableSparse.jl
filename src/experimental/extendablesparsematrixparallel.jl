mutable struct ExtendableSparseMatrixXParallel{Tm<:AbstractSparseMatrixExtension, Tv, Ti <: Integer} <: AbstractExtendableSparseMatrix{Tv, Ti}
    """
    Final matrix data
    """
    cscmatrix::SparseMatrixCSC{Tv, Ti}

    """
        Vector of dictionaries for new entries
    """
    xmatrices::Vector{Tm}

    colparts::Vector{Ti}
    partnodes::Vector{Ti}
end


function ExtendableSparseMatrixXParallel{Tm, Tv, Ti}(n,m,p::Integer) where{Tm<:AbstractSparseMatrixExtension, Tv, Ti}
    
    ExtendableSparseMatrixXParallel(spzeros(Tv, Ti, m, n),
                                    [Tm(m,n) for i=1:p],
                                    Ti[1,2],
                                    Ti[1,n+1],
                                    )
end

function partitioning!(ext::ExtendableSparseMatrixXParallel{Tm,Tv,Ti}, colparts, partnodes) where {Tm, Tv, Ti}
    ext.partnodes=partnodes
    ext.colparts=colparts
    ext
end

function ExtendableSparseMatrixXParallel{Tm, Tv, Ti}(n,m, pc::Vector) where{Tm, Tv, Ti}
    ext=ExtendableSparseMatrixXParallel(m,n,length(pc))
end


function reset!(ext::ExtendableSparseMatrixXParallel{Tm,Tv,Ti},p::Integer) where {Tm,Tv,Ti}
    m,n=size(ext.cscmatrix)
    ext.cscmatrix=spzeros(Tv, Ti, m, n)
    ext.xmatrices=[Tm(m,n) for i=1:p]
    ext.colparts=Ti[1,2]
    ext.partnodes=Ti[1,n+1]
    ext
end

function reset!(ext::ExtendableSparseMatrixXParallel)
    reset!(ext,length(ext.xmatrices))
end


function flush!(ext::ExtendableSparseMatrixXParallel{Tm,Tv,Ti}) where{Tm,Tv,Ti}
    ext.cscmatrix=Base.sum(ext.xmatrices, ext.cscmatrix)
    np=length(ext.xmatrices)
    (m,n)=size(ext.cscmatrix)
    ext.xmatrices=[Tm(m,n) for i=1:np]
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

# Needed in 1.9
function Base.:*(ext::ExtendableSparse.Experimental.ExtendableSparseMatrixXParallel{Tm, TA} where Tm<:ExtendableSparse.AbstractSparseMatrixExtension, x::Union{StridedVector, BitVector}) where TA
    mul!(similar(x),ext,x)
end

function LinearAlgebra.mul!(r, ext::ExtendableSparseMatrixXParallel, x)
    flush!(ext)
    A=ext.cscmatrix
    colparts=ext.colparts
    @show colparts
    partnodes=ext.partnodes
    @show partnodes
    rows = SparseArrays.rowvals(A)
    vals = nonzeros(A)
    r.=zero(eltype(ext))
    m,n=size(A)
    for icol=1:length(colparts)-1
        @tasks for ip=colparts[icol]:colparts[icol+1]-1
            for inode in  partnodes[ip]:partnodes[ip+1]-1
                @inbounds for i in nzrange(A,inode)
                    r[rows[i]]+=vals[i]*x[inode]
                end
            end
        end
    end
    r
end
