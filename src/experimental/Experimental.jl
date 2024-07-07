module Experimental
using ExtendableSparse, SparseArrays
using LinearAlgebra
using SparseArrays: AbstractSparseMatrixCSC
import SparseArrays: nonzeros, getcolptr,nzrange
import ExtendableSparse: flush!, reset!, rawupdateindex!, findindex
using ExtendableSparse: ColEntry, AbstractPreconditioner, @makefrommatrix, phash
using ExtendableSparse:  AbstractExtendableSparseMatrixCSC, AbstractSparseMatrixExtension
using DocStringExtensions
using Metis
using Base.Threads
using OhMyThreads: @tasks
import ExtendableSparse: factorize!, update!, partitioning!

include(joinpath(@__DIR__, "ExtendableSparseMatrixParallel", "ExtendableSparseParallel.jl"))

include(joinpath(@__DIR__, "ExtendableSparseMatrixParallel", "ilu_Al-Kurdi_Mittal.jl"))
#using .ILUAM
include(joinpath(@__DIR__, "ExtendableSparseMatrixParallel", "pilu_Al-Kurdi_Mittal.jl"))
#using .PILUAM

include(joinpath(@__DIR__, "ExtendableSparseMatrixParallel" ,"iluam.jl"))
include(joinpath(@__DIR__, "ExtendableSparseMatrixParallel", "piluam.jl"))

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



end

