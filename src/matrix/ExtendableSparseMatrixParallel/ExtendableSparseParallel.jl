include("supersparse.jl")
include("preparatory.jl")
#include("prep_time.jl")

mutable struct ExtendableSparseMatrixParallel{Tv, Ti <: Integer} <: AbstractSparseMatrix{Tv, Ti}
    """
    Final matrix data
    """
    cscmatrix::SparseMatrixCSC{Tv, Ti}

    """
    Linked list structure holding data of extension
    """
    lnkmatrices::Vector{SuperSparseMatrixLNK{Tv, Ti}}

	grid::ExtendableGrid

	nnts::Vector{Ti}
    
    sortednodesperthread::Matrix{Ti}
    
    old_noderegions::Matrix{Ti}
    
    cellsforpart::Vector{Vector{Ti}}
    
    globalindices::Vector{Vector{Ti}}
    
    new_indices::Vector{Ti}
    
    rev_new_indices::Vector{Ti}
    
    start::Vector{Ti}
    
    cellparts::Vector{Ti}
    
    nt::Ti
    
    depth::Ti

    phash::UInt64

    n::Ti

    m::Ti
    
    
end



function ExtendableSparseMatrixParallel{Tv, Ti}(nm, nt, depth; x0=0.0, x1=1.0) where {Tv, Ti <: Integer}
	grid, nnts, s, onr, cfp, gi, gc, ni, rni, starts, cellparts, depth = preparatory_multi_ps_less_reverse(nm, nt, depth, Ti; x0, x1)
	csc = spzeros(Tv, Ti, num_nodes(grid), num_nodes(grid))
	lnk = [SuperSparseMatrixLNK{Tv, Ti}(num_nodes(grid), nnts[tid]) for tid=1:nt]
	ExtendableSparseMatrixParallel{Tv, Ti}(csc, lnk, grid, nnts, s, onr, cfp, gi, ni, rni, starts, cellparts, nt, depth, phash(csc), csc.n, csc.m)
end



function addtoentry!(A::ExtendableSparseMatrixParallel{Tv, Ti}, i, j, tid, v; known_that_unknown=false) where {Tv, Ti <: Integer}
	if known_that_unknown
		A.lnkmatrices[tid][i, A.sortednodesperthread[tid, j]] += v
		return
	end
	
	if updatentryCSC2!(A.cscmatrix, i, j, v)
	else
		A.lnkmatrices[tid][i, A.sortednodesperthread[tid, j]] += v
	end
end


#=
function addtoentry!(A::ExtendableSparseMatrixParallel{Tv, Ti}, i, j, v; known_that_unknown=false) where {Tv, Ti <: Integer}
	if known_that_unknown
		level, tid = last_nz(ext.old_noderegions[:, ext.rev_new_indices[j]])
		A.lnkmatrices[tid][i, A.sortednodesperthread[tid, j]] += v
		return
	end
	
	if updatentryCSC2!(A.cscmatrix, i, j, v)
	else
		level, tid = last_nz(ext.old_noderegions[:, ext.rev_new_indices[j]])
		A.lnkmatrices[tid][i, A.sortednodesperthread[tid, j]] += v
	end
end
=#


"""
`function addtoentry!(A::ExtendableSparseMatrixParallel{Tv, Ti}, i, j, v; known_that_unknown=true) where {Tv, Ti <: Integer}`

A[i,j] += v, using any partition.
If the partition should be specified (for parallel use), use 
`function addtoentry!(A::ExtendableSparseMatrixParallel{Tv, Ti}, i, j, tid, v; known_that_unknown=true) where {Tv, Ti <: Integer}`.
"""
function addtoentry!(A::ExtendableSparseMatrixParallel{Tv, Ti}, i, j, v; known_that_unknown=false) where {Tv, Ti <: Integer}
	if known_that_unknown
		level, tid = last_nz(A.old_noderegions[:, A.rev_new_indices[j]])
		A.lnkmatrices[tid][i, A.sortednodesperthread[tid, j]] += v
		return
	end
	
	if updatentryCSC2!(A.cscmatrix, i, j, v)
	else
		level, tid = last_nz(A.old_noderegions[:, A.rev_new_indices[j]])
		A.lnkmatrices[tid][i, A.sortednodesperthread[tid, j]] += v
	end
end

#---------------------------------


function updateindex!(ext::ExtendableSparseMatrixParallel{Tv, Ti},
                      op,
                      v,
                      i,
                      j) where {Tv, Ti <: Integer}
    k = ExtendableSparse.findindex(ext.cscmatrix, i, j)
    if k > 0
        ext.cscmatrix.nzval[k] = op(ext.cscmatrix.nzval[k], v)
        return
    else
    	level, tid = last_nz(ext.old_noderegions[:, ext.rev_new_indices[j]])
		updateindex!(ext.lnkmatrices[tid], op, v, i, ext.sortednodesperthread[tid, j])
    end
    ext
end

function updateindex!(ext::ExtendableSparseMatrixParallel{Tv, Ti},
                      op,
                      v,
                      i,
                      j,
                      tid) where {Tv, Ti <: Integer}
    k = ExtendableSparse.findindex(ext.cscmatrix, i, j)
    if k > 0
        ext.cscmatrix.nzval[k] = op(ext.cscmatrix.nzval[k], v)
        return
    else
    	updateindex!(ext.lnkmatrices[tid], op, v, i, ext.sortednodesperthread[tid, j])
    end
    ext
end

function rawupdateindex!(ext::ExtendableSparseMatrixParallel{Tv, Ti},
                         op,
                         v,
                         i,
                         j) where {Tv, Ti <: Integer}
    k = ExtendableSparse.findindex(ext.cscmatrix, i, j)
    if k > 0
        ext.cscmatrix.nzval[k] = op(ext.cscmatrix.nzval[k], v)
    else
        level, tid = last_nz(ext.old_noderegions[:, ext.rev_new_indices[j]])
	    rawupdateindex!(ext.lnkmatrices[tid], op, v, i, ext.sortednodesperthread[tid, j])
    end
    ext
end

function rawupdateindex!(ext::ExtendableSparseMatrixParallel{Tv, Ti},
                         op,
                         v,
                         i,
                         j, 
                         tid) where {Tv, Ti <: Integer}
    k = ExtendableSparse.findindex(ext.cscmatrix, i, j)
    if k > 0
        ext.cscmatrix.nzval[k] = op(ext.cscmatrix.nzval[k], v)
    else
        rawupdateindex!(ext.lnkmatrices[tid], op, v, i, ext.sortednodesperthread[tid, j])
    end
    ext
end

function Base.getindex(ext::ExtendableSparseMatrixParallel{Tv, Ti},
                       i::Integer,
                       j::Integer) where {Tv, Ti <: Integer}
    k = ExtendableSparse.findindex(ext.cscmatrix, i, j)
    if k > 0
        return ext.cscmatrix.nzval[k]
    end
    
    level, tid = last_nz(ext.old_noderegions[:, ext.rev_new_indices[j]])
	ext.lnkmatrices[tid][i, ext.sortednodesperthread[tid, j]]
    
end

function Base.setindex!(ext::ExtendableSparseMatrixParallel{Tv, Ti},
                        v::Union{Number,AbstractVecOrMat},
                        i::Integer,
                        j::Integer) where {Tv, Ti}
    k = ExtendableSparse.findindex(ext.cscmatrix, i, j)
    if k > 0
        ext.cscmatrix.nzval[k] = v
    else
		level, tid = last_nz(ext.old_noderegions[:, ext.rev_new_indices[j]])
		#@info typeof(tid), typeof(j)
		jj = ext.sortednodesperthread[tid, j]
		ext.lnkmatrices[tid][i, jj] = v
    end
end



#------------------------------------

function reset!(A::ExtendableSparseMatrixParallel{Tv, Ti}) where {Tv, Ti <: Integer}
	A.cscmatrix = spzeros(Tv, Ti, num_nodes(A.grid), num_nodes(A.grid))
	A.lnkmatrices = [SuperSparseMatrixLNK{Tv, Ti}(num_nodes(A.grid), A.nnts[tid]) for tid=1:A.nt]
end

function nnz_flush(ext::ExtendableSparseMatrixParallel)
    flush!(ext)
    return nnz(ext.cscmatrix)
end

function nnz_noflush(ext::ExtendableSparseMatrixParallel)
    return nnz(ext.cscmatrix), sum([ext.lnkmatrices[i].nnz for i=1:ext.nt])
end
	
function matrixindextype(A::ExtendableSparseMatrixParallel{Tv, Ti}) where {Tv, Ti <: Integer}
	Ti
end

function matrixvaluetype(A::ExtendableSparseMatrixParallel{Tv, Ti}) where {Tv, Ti <: Integer}
	Tv
end



function Base.show(io::IO, ::MIME"text/plain", ext::ExtendableSparseMatrixParallel)
    #flush!(ext)
    xnnzCSC, xnnzLNK = nnz_noflush(ext)
    m, n = size(ext)
    print(io,
          m,
          "×",
          n,
          " ",
          typeof(ext),
          " with ",
          xnnzCSC,
          " stored ",
          xnnzCSC == 1 ? "entry" : "entries",
          " in CSC and ",
          xnnzLNK,
          " stored ",
          xnnzLNK == 1 ? "entry" : "entries",
          " in LNK. CSC:")

    if !haskey(io, :compact)
        io = IOContext(io, :compact => true)
    end

    if !(m == 0 || n == 0 || xnnzCSC == 0)
        print(io, ":\n")
        Base.print_array(IOContext(io), ext.cscmatrix)
    end
end

"""
`function entryexists2(CSC, i, j)`

Find out if CSC already has an nonzero entry at i,j without any allocations
"""
function entryexists2(CSC, i, j) #find out if CSC already has an nonzero entry at i,j
	#vals = 
	#ids = CSC.colptr[j]:(CSC.colptr[j+1]-1)
	i in view(CSC.rowval, CSC.colptr[j]:(CSC.colptr[j+1]-1))
end


function updatentryCSC2!(CSC::SparseArrays.SparseMatrixCSC{Tv, Ti}, i::Integer, j::Integer, v) where {Tv, Ti <: Integer}
	p1 = CSC.colptr[j]
	p2 = CSC.colptr[j+1]-1

	searchk = searchsortedfirst(view(CSC.rowval, p1:p2), i) + p1 - 1
	
	if (searchk <= p2) && (CSC.rowval[searchk] == i)
		CSC.nzval[searchk] += v
		return true
	else
		return false
	end
end

Base.size(A::ExtendableSparseMatrixParallel) = (A.cscmatrix.m, A.cscmatrix.n)

include("struct_flush.jl")
