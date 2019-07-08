module ExtendableSparse
using SparseArrays

mutable struct SparseMatrixExtension{Tv,Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti}
    m::Ti
    n::Ti
    nnz::Ti
    colptr::Vector{Ti}      
    rowval::Vector{Ti}      
    nzval::Vector{Tv}       
    
    function SparseMatrixExtension{Tv,Ti}(m,n) where {Tv,Ti<:Integer}
        colptr=zeros(Ti,n)
        rowval=zeros(Ti,n)
        nzval=zeros(Tv,n)
        new(m,n,0,zeros(Ti,n),zeros(Ti,n),zeros(Tv,n))
    end
end

function Base.setindex!(E::SparseMatrixExtension{Tv,Ti}, _v, _i::Integer, _j::Integer) where {Tv,Ti<:Integer}
    v = convert(Tv, _v)
    i = convert(Ti, _i)
    j = convert(Ti, _j)

    if !((1 <= i <= E.m) & (1 <= j <= E.n))
        throw(BoundsError(E, (i,j)))
    end
    
    if iszero(v)
        return E
    end
    
    if E.rowval[j]==0
        E.rowval[j]=i       
        E.nzval[j]=v
        E.nnz+=1
        return E
    end

    k=j
    k0=j
    while k>0
        if E.rowval[k]==i
            E.nzval[k]=v
            return E
        end
        k0=k
        k=E.colptr[k]
    end
    push!(E.nzval,v)
    push!(E.rowval,i)
    push!(E.colptr,-1)
    E.colptr[k0]=length(E.nzval)
    E.nnz+=1
    return E
end

function Base.getindex(E::SparseMatrixExtension{Tv,Ti},i::Integer, j::Integer) where {Tv,Ti<:Integer}
    if !((1 <= i <= E.m) & (1 <= j <= E.n))
        throw(BoundsError(E, (i,j)))
    end

    k=j
    while k>0
        if E.rowval[k]==i
            return E.nzval[k]
        end
        k=E.colptr[k]
    end
    return zero(Tv)
end

Base.size(E::SparseMatrixExtension) = (E.m, E.n)
SparseArrays.nnz(E::SparseMatrixExtension)=E.nnz



#####################################################################################################
mutable struct ExtendableSparseMatrixCSC{Tv,Ti<:Integer} <: AbstractSparseMatrix{Tv,Ti}
    m::Int
    n::Int
    cscmatrix::SparseMatrixCSC{Tv,Ti}
    extmatrix::SparseMatrixExtension{Tv,Ti}
end

function Base.setindex!(m::ExtendableSparseMatrixCSC, v, i::Integer, j::Integer)
    setindex!(m.cscmatrix,v,i,j)
end

function Base.getindex(m::ExtendableSparseMatrixCSC,i::Integer, j::Integer)
    return getindex(m.cscmatrix,i,j)
end

struct ColEntry{Tv,Ti<:Integer}
    j::Ti
    v::Tv
end

isless(x::ColEntry,y::ColEntry)=(x.i<y.i)

function splice(E::ExtendableSparseMatrixCSC,S::SparseMatrixCSC)


    nnz=nnz(S)+nnz(E)
    colptr=Vector{Ti}(undef,S.m+1)
    rowval=Vector{Ti}(undef,nnz)
    nzval=Vector{Tv}(undef,nnz)
    
    E_maxcol_ext=0
    S_maxcol=0
    for j=1:m
        lrow=0
        k=j
        while k>0
            lrow+=1
            k=E.colptr[k]
        end
        E_maxcol_ext=max(lrow,E_maxcol)
        S_maxcol=max(S.colptr[j+1]-S.colptr[j],S_maxcol)
    end
    
    col=Vector{ColEntry}(undef,E_maxcol+S_Maxcol+10)
    maxcol=0

    i=1
    for j=1:m
        # put extension entries into row and sort them
        k=j
        while k>0
            if colptr[k]>0
                lxcol+=1
                col[lxcol].j=colptr[k]
                col[lxcol].v=nzval[k]
                k=colptr[k]
            end
        end

        sort!(col,lt=isless)
        # jointly sort old and mew entries into colptr

        i0=i
        colptr[j]=i
        jcol=0
        k=S.colptr[j]
        while true
            if k<S.colptr[j+1] && icol>lxcol || S.colptr[k]<col[jcol].j
                rowval[i]=S.rowval[k]
                nzval[i]=S.nzval[k]
                k+=1
                i+=1
                continue
            end
            if jcol<lxcol
                rowval[i]=col[jcol].j
                nzval[i]=col[jcol].v
                jcol+=1
                i++
                continue
            end
            break
        end
    end
    maxrow=max(maxrow,i-i0)
    colptr[j]=i
    return SparseMatrixCSC(m,n,colptr,rowval,nzval)
end




export SparseMatrixExtension,ExtendableSparseMatrixCSC
end # module
