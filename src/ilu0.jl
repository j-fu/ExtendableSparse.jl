mutable struct ILU0Preconditioner{Tv, Ti} <: AbstractPreconditioner{Tv,Ti}
    A::ExtendableSparseMatrix{Tv,Ti}
    xdiag::Array{Tv,1}
    idiag::Array{Ti,1}
    phash::UInt64
    function ILU0Preconditioner{Tv,Ti}() where {Tv,Ti}
        p=new()
        p.phash=0
        p
    end
end

"""
```
ILU0Preconditioner(;valuetype=Float64,indextype=Int64)
ILU0Preconditioner(matrix)
```

ILU preconditioner with zero fill-in.
"""
ILU0Preconditioner(;valuetype::Type=Float64, indextype::Type=Int64)=ILU0Preconditioner{valuetype,indextype}()


function update!(precon::ILU0Preconditioner{Tv,Ti}) where {Tv,Ti}
    flush!(precon.A)
    cscmatrix=precon.A.cscmatrix
    colptr=cscmatrix.colptr
    rowval=cscmatrix.rowval
    nzval=cscmatrix.nzval
    n=cscmatrix.n

    if precon.phash==0
        n=size(precon.A,1)
        precon.xdiag=Array{Tv,1}(undef,n)
        precon.idiag=Array{Ti,1}(undef,n)
    end

    xdiag=precon.xdiag
    idiag=precon.idiag


    
    # Find main diagonal index and
    # copy main diagonal values
    if precon.phash != precon.A.phash
        @inbounds for j=1:n
            @inbounds for k=colptr[j]:colptr[j+1]-1
                i=rowval[k]
                if i==j
                    idiag[j]=k
                    break
                end
            end
        end
        precon.phash=precon.A.phash
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


function  LinearAlgebra.ldiv!(u::AbstractArray{T,1}, precon::ILU0Preconditioner{Tv,Ti}, v::AbstractArray{T,1}) where {T,Tv,Ti}
    cscmatrix=precon.A.cscmatrix
    colptr=cscmatrix.colptr
    rowval=cscmatrix.rowval
    n=cscmatrix.n
    nzval=cscmatrix.nzval
    xdiag=precon.xdiag
    idiag=precon.idiag
    
    @inbounds for j=1:n
        x=zero(Tv)
        @inbounds for k=colptr[j]:idiag[j]-1
            x+=nzval[k]*u[rowval[k]]
        end
        u[j]=xdiag[j]*(v[j]-x)
    end
    
    @inbounds for j=n:-1:1
        x=zero(Tv)
        @inbounds for k=idiag[j]+1:colptr[j+1]-1
            x+=u[rowval[k]]*nzval[k]
        end
        u[j]-=x*xdiag[j]
    end
end


function LinearAlgebra.ldiv!(precon::ILU0Preconditioner, v::AbstractArray{T,1} where T)
    ldiv!(v, precon, v)
end


