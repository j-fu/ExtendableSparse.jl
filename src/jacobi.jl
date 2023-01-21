mutable struct JacobiPreconditioner{Tv, Ti} <: AbstractPreconditioner{Tv, Ti}
    A::ExtendableSparseMatrix{Tv, Ti}
    invdiag::Array{Tv, 1}
    function JacobiPreconditioner{Tv, Ti}() where {Tv, Ti}
        p = new()
        p.invdiag = zeros(Tv, 0)
        p
    end
end

"""
```
JacobiPreconditioner(;valuetype=Float64, indextype=Int64)
JacobiPreconditioner(matrix)
```

Jacobi preconditioner
"""
function JacobiPreconditioner(; valuetype::Type = Float64, indextype::Type = Int64)
    JacobiPreconditioner{valuetype, indextype}()
end

function update!(precon::JacobiPreconditioner{Tv, Ti}) where {Tv, Ti}
    flush!(precon.A)
    cscmatrix = precon.A.cscmatrix
    if length(precon.invdiag) == 0
        precon.invdiag = Array{Tv, 1}(undef, cscmatrix.n)
    end
    invdiag = precon.invdiag
    n = cscmatrix.n
    @inbounds for i = 1:n
        invdiag[i] = 1.0 / cscmatrix[i, i]
    end
    precon
end

function LinearAlgebra.ldiv!(u::AbstractArray{T, 1} where {T},
                             precon::JacobiPreconditioner,
                             v::AbstractArray{T, 1} where {T})
    invdiag = precon.invdiag
    n = length(invdiag)
    @inbounds @simd for i = 1:n
        u[i] = invdiag[i] * v[i]
    end
end

function LinearAlgebra.ldiv!(precon::JacobiPreconditioner, v::AbstractArray{T, 1} where {T})
    ldiv!(v, precon, v)
end
