abstract type AbstractPardisoLU{Tv,Ti} <: AbstractLUFactorization{Tv,Ti} end

mutable struct PardisoLU{Tv, Ti} <: AbstractPardisoLU{Tv,Ti}
    A::Union{ExtendableSparseMatrix{Tv,Ti},Nothing}
    ps::Pardiso.PardisoSolver
    phash::UInt64
end

function PardisoLU{Tv,Ti}() where {Tv,Ti}
    fact=PardisoLU{Tv,Ti}(nothing,Pardiso.PardisoSolver(),0)
    set_default_matrixtype!(fact)
end

"""
```
PardisoLU(;valuetype=Float64, indextype=Int64)
PardisoLU(matrix)
```

LU factorization based on pardiso. For using this, you need to issue `using Pardiso`
and have the pardiso library from  [pardiso-project.org](https://pardiso-project.org) 
[installed](https://github.com/JuliaSparse/Pardiso.jl#pardiso-60).

For (setting Pardiso internal parameters)[https://github.com/JuliaSparse/Pardiso.jl#readme], 
you can access the `PardisoSolver` e.g. like
```
using Pardiso
plu=PardisoLU()
Pardiso.set_iparm!(plu.ps,5,13.0)
```
"""
PardisoLU(;valuetype::Type=Float64, indextype::Type=Int64)=PardisoLU{valuetype,indextype}()



#############################################################################################
mutable struct MKLPardisoLU{Tv, Ti} <: AbstractPardisoLU{Tv,Ti}
    A::Union{ExtendableSparseMatrix{Tv,Ti},Nothing}
    ps::Pardiso.MKLPardisoSolver
    phash::UInt64
end

function MKLPardisoLU{Tv,Ti}() where {Tv,Ti}
    fact=MKLPardisoLU{Tv,Ti}(nothing,Pardiso.MKLPardisoSolver(),0)
    set_default_matrixtype!(fact)
end


"""
```
MKLPardisoLU(;valuetype=Float64, indextype=Int64)
MKLPardisoLU(matrix)
```

LU factorization based on pardiso. For using this, you need to issue `using Pardiso`.
This version  uses the early 2000's fork in Intel's MKL library.


For (setting Pardiso internal parameters)[https://github.com/JuliaSparse/Pardiso.jl#readme], 
you can access the `PardisoSolver` e.g. like
```
using Pardiso
plu=MKLPardisoLU()
Pardiso.set_iparm!(plu.ps,5,13.0)
```
"""
MKLPardisoLU(;valuetype::Type=Float64, indextype::Type=Int64)=MKLPardisoLU{valuetype,indextype}()


##########################################################################################
function set_default_matrixtype!(fact::AbstractPardisoLU{Tv,Ti}) where {Tv, Ti}
    if Tv<:Complex
        Pardiso.set_matrixtype!(fact.ps,Pardiso.COMPLEX_NONSYM)
    else
        Pardiso.set_matrixtype!(fact.ps,Pardiso.REAL_NONSYM)
    end
    fact
end

function update!(lufact::AbstractPardisoLU{Tv,Ti}) where {Tv, Ti}
    ps=lufact.ps
    flush!(lufact.A)
    Acsc=lufact.A.cscmatrix
    if lufact.phash!=lufact.A.phash 
        Pardiso.pardisoinit(ps)
        Pardiso.set_phase!(ps, Pardiso.RELEASE_ALL)
        Pardiso.pardiso(ps, Tv[], Acsc, Tv[])
        Pardiso.set_phase!(ps, Pardiso.ANALYSIS_NUM_FACT)
        lufact.phash=lufact.A.phash
    else
        Pardiso.set_phase!(ps, Pardiso.NUM_FACT)
    end
    Pardiso.fix_iparm!(ps, :N)
    Pardiso.pardiso(ps, Tv[], Acsc, Tv[])
    lufact
end

function LinearAlgebra.ldiv!(u::AbstractArray{T,1} where T, lufact::AbstractPardisoLU, v::AbstractArray{T,1} where T)
    ps=lufact.ps
    Acsc=lufact.A.cscmatrix
    Pardiso.set_phase!(ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    Pardiso.pardiso(ps, u, Acsc, v)
    u
end

LinearAlgebra.ldiv!(fact::AbstractPardisoLU, v::AbstractArray{T,1} where T)=ldiv!(v,fact,copy(v))

@eval begin
    @makefrommatrix PardisoLU
    @makefrommatrix MKLPardisoLU
end
