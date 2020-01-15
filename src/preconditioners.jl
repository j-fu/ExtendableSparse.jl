abstract type AbstractPreconditioner{Tv,Ti} end

need_symbolic_update(precon::AbstractPreconditioner)=precon.extmatrix.pattern_timestamp>precon.pattern_timestamp
timestamp!(precon::AbstractPreconditioner)= precon.pattern_timestamp=time()

include("jacobi.jl")
include("ilu0.jl")
