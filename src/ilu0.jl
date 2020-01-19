mutable struct ILU0Preconditioner{Tv, Ti} <: AbstractExtendablePreconditioner{Tv,Ti}
    extmatrix::ExtendableSparseMatrix{Tv,Ti}
    xdiag::Array{Tv,1}
    idiag::Array{Ti,1}
    pattern_timestamp::Float64
end


function ILU0Preconditioner(extmatrix::ExtendableSparseMatrix{Tv,Ti}) where {Tv,Ti}
    @assert size(extmatrix,1)==size(extmatrix,2)
    flush!(extmatrix)
    n=size(extmatrix,1)
    xdiag=Array{Tv,1}(undef,n)
    idiag=Array{Ti,1}(undef,n)
    precon=ILU0Preconditioner{Tv, Ti}(extmatrix,xdiag,idiag,0.0)
    update!(precon)
end

ILU0Preconditioner(cscmatrix::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}=ILU0Preconditioner(ExtendableSparseMatrix(cscmatrix))


function update!(precon::ILU0Preconditioner{Tv,Ti}) where {Tv,Ti}
    cscmatrix=precon.extmatrix.cscmatrix
    colptr=cscmatrix.colptr
    rowval=cscmatrix.rowval
    nzval=cscmatrix.nzval
    n=cscmatrix.n
    xdiag=precon.xdiag
    idiag=precon.idiag

    # Find main diagonal index and
    # copy main diagonal values
    if need_symbolic_update(precon)
        @inbounds for j=1:n
            @inbounds for k=colptr[j]:colptr[j+1]-1
                i=rowval[k]
                if i==j
                    idiag[j]=k
                    break
                end
            end
        end
        timestamp!(precon)
    end
    
    @inbounds for j=1:n
        xdiag[j]=one(Tv)/nzval[idiag[j]]
        @inbounds for k=idiag[j]+1:colptr[j+1]-1
            i=rowval[k]
            for l=colptr[i]:colptr[i+1]-1
                if rowval[l]==j
                    xdiag[i]-=nzval[l]*xdiag[j]*nzval[k]
                    break
                end
            end
        end
    end
    precon
end


function  LinearAlgebra.ldiv!(u::AbstractArray{T,1}, precon::ILU0Preconditioner, v::AbstractArray{T,1}) where T
    cscmatrix=precon.extmatrix.cscmatrix
    colptr=cscmatrix.colptr
    rowval=cscmatrix.rowval
    n=cscmatrix.n
    nzval=cscmatrix.nzval
    xdiag=precon.xdiag
    idiag=precon.idiag
    
    @inbounds for j=1:n
        x=zero(T)
        @inbounds for k=colptr[j]:idiag[j]-1
            x+=nzval[k]*u[rowval[k]]
        end
        u[j]=xdiag[j]*(v[j]-x)
    end
    
    @inbounds for j=n:-1:1
        x=zero(T)
        @inbounds for k=idiag[j]+1:colptr[j+1]-1
            x+=u[rowval[k]]*nzval[k]
        end
        u[j]-=x*xdiag[j]
    end
end


