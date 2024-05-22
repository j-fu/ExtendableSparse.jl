module Experimental
using ExtendableSparse, SparseArrays
using LinearAlgebra
using SparseArrays: AbstractSparseMatrixCSC
import SparseArrays: nonzeros, getcolptr,nzrange
import ExtendableSparse: flush!, reset!, rawupdateindex!, findindex
using ExtendableSparse: ColEntry, AbstractPreconditioner, @makefrommatrix, phash
using DocStringExtensions
using Metis
using Base.Threads
using OhMyThreads: @tasks


include(joinpath(@__DIR__, "..", "matrix", "ExtendableSparseMatrixParallel", "ExtendableSparseParallel.jl"))

include(joinpath(@__DIR__, "..", "factorizations","ilu_Al-Kurdi_Mittal.jl"))
#using .ILUAM
include(joinpath(@__DIR__, "..", "factorizations","pilu_Al-Kurdi_Mittal.jl"))
#using .PILUAM

include(joinpath(@__DIR__, "..", "factorizations","iluam.jl"))
include(joinpath(@__DIR__, "..", "factorizations","piluam.jl"))

@eval begin
    @makefrommatrix ILUAMPreconditioner
    @makefrommatrix PILUAMPreconditioner
end

function factorize!(p::PILUAMPreconditioner, A::ExtendableSparseMatrixParallel)
    p.A = A
    update!(p)
    p
end
                
export ExtendableSparseMatrixParallel, SuperSparseMatrixLNK
export addtoentry!, reset!, dummy_assembly!, preparatory_multi_ps_less_reverse, fr, addtoentry!,  compare_matrices_light
export     ILUAMPreconditioner,    PILUAMPreconditioner
export     reorderlinsys, nnz_noflush


include("abstractextendable.jl")

include("sparsematrixdict.jl")
export SparseMatrixDict

include("extendablesparsematrixdict.jl")
export ExtendableSparseMatrixDict

include("extendablesparsematrixparalleldict.jl")
export ExtendableSparseMatrixParallelDict, partcolors!


include("parallel_testtools.jl")
export part2d, showgrid, partassemble!,  assemblepartition!

end

