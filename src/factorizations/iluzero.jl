mutable struct ILUZeroPreconditioner <: AbstractPreconditioner
    A::ExtendableSparseMatrix
    factorization::ILUZero.ILU0Precon
    phash::UInt64
    function ILUZeroPreconditioner()
        p = new()
        p.phash = 0
        p
    end
end

"""
```
ILUZeroPreconditioner()
ILUZeroPreconditioner(matrix)
```
Incomplete LU preconditioner with zero fill-in using  [ILUZero.jl](https://github.com/mcovalt/ILUZero.jl). This preconditioner
also calculates and stores updates to the off-diagonal entries and thus delivers better convergence than  the [`ILU0Preconditioner`](@ref).
"""
function ILUZeroPreconditioner end

function update!(p::ILUZeroPreconditioner)
    flush!(p.A)
    if p.A.phash != p.phash
        p.factorization = ILUZero.ilu0(p.A.cscmatrix)
        p.phash=p.A.phash
    else
        ILUZero.ilu0!(p.factorization, p.A.cscmatrix)
    end
    p
end

allow_views(::ILUZeroPreconditioner)=true
allow_views(::Type{ILUZeroPreconditioner})=true



mutable struct PointBlockILUZeroPreconditioner <: AbstractPreconditioner
    A::ExtendableSparseMatrix
    factorization::ILUZero.ILU0Precon
    phash::UInt64
    blocksize::Int
    function PointBlockILUZeroPreconditioner(;blocksize=1)
        p = new()
        p.phash = 0
        p.blocksize=blocksize
        p
    end
end

"""
```
PointBlockILUZeroPreconditioner(;blocksize)
PointBlockILUZeroPreconditioner(matrix;blocksize)
```
Incomplete LU preconditioner with zero fill-in using  [ILUZero.jl](https://github.com/mcovalt/ILUZero.jl). This preconditioner
also calculates and stores updates to the off-diagonal entries and thus delivers better convergence than  the [`ILU0Preconditioner`](@ref).
"""
function PointBlockILUZeroPreconditioner end

function update!(p::PointBlockILUZeroPreconditioner)
    flush!(p.A)
    Ab=pointblock(p.A.cscmatrix,p.blocksize)
    if p.A.phash != p.phash
        p.factorization = ILUZero.ilu0(Ab, SVector{p.blocksize,eltype(p.A)})
        p.phash=p.A.phash
    else
        ILUZero.ilu0!(p.factorization, Ab)
    end
    p
end


function LinearAlgebra.ldiv!(p::PointBlockILUZeroPreconditioner,v)
    vv=reinterpret(SVector{p.blocksize,eltype(v)},v)
    LinearAlgebra.ldiv!(vv,p.factorization,vv)
    v
end    

function LinearAlgebra.ldiv!(u,p::PointBlockILUZeroPreconditioner,v)
    LinearAlgebra.ldiv!(reinterpret(SVector{p.blocksize,eltype(u)},u),
                        p.factorization,
                        reinterpret(SVector{p.blocksize,eltype(v)},v)
                        )
    u
end


