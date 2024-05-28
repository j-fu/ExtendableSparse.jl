module Experimental
using ExtendableSparse, SparseArrays
using LinearAlgebra
using SparseArrays: AbstractSparseMatrixCSC
import SparseArrays: nonzeros, getcolptr,nzrange
import ExtendableSparse: flush!, reset!, rawupdateindex!, findindex
using ExtendableSparse: ColEntry, AbstractPreconditioner, @makefrommatrix, phash
using ExtendableSparse:  AbstractExtendableSparseMatrix, AbstractSparseMatrixExtension
using DocStringExtensions
using Metis
using Base.Threads
using OhMyThreads: @tasks
import ExtendableSparse: factorize!, update!


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


include("sparsematrixdict.jl")
export SparseMatrixDict

include("sparsematrixlnkx.jl")
export SparseMatrixLNKX

include("sparsematrixlnkdict.jl")
export SparseMatrixLNKDict

include("extendablesparsematrixscalar.jl")
export ExtendableSparseMatrixScalar

const ExtendableSparseMatrixDict{Tv,Ti}=ExtendableSparseMatrixScalar{SparseMatrixDict{Tv,Ti},Tv,Ti}
export ExtendableSparseMatrixDict


const ExtendableSparseMatrixLNKDict{Tv,Ti}=ExtendableSparseMatrixScalar{SparseMatrixLNKDict{Tv,Ti},Tv,Ti}
export ExtendableSparseMatrixLNKDict

const ExtendableSparseMatrixLNKX{Tv,Ti}=ExtendableSparseMatrixScalar{SparseMatrixLNKX{Tv,Ti},Tv,Ti}
export ExtendableSparseMatrixLNKX

const ExtendableSparseMatrixLNK{Tv,Ti}=ExtendableSparseMatrixScalar{SparseMatrixLNK{Tv,Ti},Tv,Ti}
export ExtendableSparseMatrixLNK


include("extendablesparsematrixparallel.jl")
const ExtendableSparseMatrixParallelDict{Tv,Ti}=ExtendableSparseMatrixXParallel{SparseMatrixDict{Tv,Ti},Tv,Ti}
ExtendableSparseMatrixParallelDict(m,n,p)= ExtendableSparseMatrixParallelDict{Float64,Int64}(m,n,p)
export ExtendableSparseMatrixParallelDict, partcolors!

const ExtendableSparseMatrixParallelLNKX{Tv,Ti}=ExtendableSparseMatrixXParallel{SparseMatrixLNKX{Tv,Ti},Tv,Ti}
ExtendableSparseMatrixParallelLNKX(m,n,p)= ExtendableSparseMatrixParallelLNKX{Float64,Int64}(m,n,p)
export ExtendableSparseMatrixParallelLNKX

const ExtendableSparseMatrixParallelLNKDict{Tv,Ti}=ExtendableSparseMatrixXParallel{SparseMatrixLNKDict{Tv,Ti},Tv,Ti}
ExtendableSparseMatrixParallelLNKDict(m,n,p)= ExtendableSparseMatrixParallelLNKDict{Float64,Int64}(m,n,p)
export ExtendableSparseMatrixParallelLNKDict


include("parallel_testtools.jl")
export part2d, showgrid, partassemble!,  assemblepartition!

end

