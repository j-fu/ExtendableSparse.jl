include("supersparse.jl")
include("preparatory.jl")
#include("prep_time.jl")

mutable struct ExtendableSparseMatrixParallel{Tv, Ti <: Integer} <: AbstractSparseMatrix{Tv, Ti}
    """
    Final matrix data
    """
    cscmatrix::SparseMatrixCSC{Tv, Ti}

    """
    Linked list structures holding data of extension, one for each thread
    """
    lnkmatrices::Vector{SuperSparseMatrixLNK{Tv, Ti}}

    """
    Number of Nodes per Threads
    """
	nnts::Vector{Ti}
    
    """
    sortednodesperthread[i,j] = local index of the j-th global column in the i-th LNK matrix
    (this is used e.g. when assembling the matrix)
    """
    sortednodesperthread::Matrix{Ti}
    
    """
    depth+1 x nn matrix,
    old_noderegions[i,j] = region in which node j is, in level i 
    old refers to the fact that j is the 'old index' (i.e. grid index, not matrix index, see 'new_indices')
    """
    old_noderegions::Matrix{Ti}
    
    """
    cellsforpart[i] is a vector containing all cells in the i-th region
    cellsforpart has length nt*depth + 1
    """
    cellsforpart::Vector{Vector{Ti}}
    
    """
    globalindices[i][j] = index in the global (ESMP & CSC) matrix of the j-th column of the i-th LNK matrix
    (this maps the local indices (in the LNKs) to the global indices (ESMP & CSC)) 
    """
    globalindices::Vector{Vector{Ti}}
    
    """
    For some applications such as the parallel ILU preconditioner, a block form is necessary.
    Thus, the columns are reordered and the A[i,i] does not correspond to the i-th node of the grid,
    but A[new_indices[i], new_indices[i]] does
    """
    new_indices::Vector{Ti}
    
    """
    Reverse: rev_new_indices[new_indices[i]] = i, for all i
    """
    rev_new_indices::Vector{Ti}
    
    """
    starts[i] gives the first column of the i-th region, i.e. starts[1] = 1
    starts has length nt*depth + 1
    """
    start::Vector{Ti}

    """
    cellparts[i] = region of the i-th cell
    """
    cellparts::Vector{Ti}
    
    """
    Number of threads
    """
    nt::Ti
    
    """
    How often is the separator partitioned? (if never: depth = 1)
    """
    depth::Ti

    phash::UInt64

    """
    Number of rows / number of nodes in grid 
    """
    n::Ti

    """
    Number of columns / number of nodes in grid (only works for square matrices)
    """
    m::Ti
    
    
end


"""
$(SIGNATURES)

`ExtendableSparseMatrixParallel{Tv, Ti}(mat_cell_node, nc, nn, nt, depth; block_struct = true) where {Tv, Ti <: Integer}`

Create an ExtendableSparseMatrixParallel based on a grid.
The grid is specified by nc (number of cells), nn (number of nodes) and the `mat_cell_node` (i.e. grid[CellNodes] if ExtendableGrids is used). 
Here, `mat_cell_node[k,i]` is the i-th node in the k-th cell.
The matrix structure is made for parallel computations with `nt` threads.
`depth` is the number of partition layers, for depth=1, there are nt parts and 1 separator, for depth=2, the separator is partitioned again
`block_struct=true` means, the matrix should be reordered two have a block structure, this is necessary for parallel ILU, for `false`, the matrix is not reordered 
"""
function ExtendableSparseMatrixParallel{Tv, Ti}(mat_cell_node, nc, nn, nt, depth; block_struct = true) where {Tv, Ti <: Integer}
	nnts, s, onr, cfp, gi, ni, rni, starts, cellparts, depth = preparatory_multi_ps_less_reverse(mat_cell_node, nc, nn, nt, depth, Ti; block_struct)
	csc = spzeros(Tv, Ti, nn, nn)
	lnk = [SuperSparseMatrixLNK{Tv, Ti}(nn, nnts[tid]) for tid=1:nt]
	ExtendableSparseMatrixParallel{Tv, Ti}(csc, lnk, nnts, s, onr, cfp, gi, ni, rni, starts, cellparts, nt, depth, phash(csc), csc.n, csc.m)
end


"""
$(SIGNATURES)

`addtoentry!(A::ExtendableSparseMatrixParallel{Tv, Ti}, i, j, tid, v; known_that_unknown=false) where {Tv, Ti <: Integer}`

`A[i,j] += v`
This function should be used, if the thread in which the entry appears is known (`tid`).
If the thread is not known, use `addtoentry!(A::ExtendableSparseMatrixParallel{Tv, Ti}, i, j, v; known_that_unknown=false)`, this function calculates `tid`.
If you know that the entry is not yet known to the CSC structure, set `known_that_unknown=true`.
"""
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


"""
$(SIGNATURES)

`addtoentry!(A::ExtendableSparseMatrixParallel{Tv, Ti}, i, j, v; known_that_unknown=false) where {Tv, Ti <: Integer}`

A[i,j] += v, using any partition.
If the partition should be specified (for parallel use), use 
`function addtoentry!(A::ExtendableSparseMatrixParallel{Tv, Ti}, i, j, tid, v; known_that_unknown=false) where {Tv, Ti <: Integer}`.
"""
function addtoentry!(A::ExtendableSparseMatrixParallel{Tv, Ti}, i, j, v; known_that_unknown=false) where {Tv, Ti <: Integer}
	if known_that_unknown
		level, tid = last_nz(A.old_noderegions[:, A.rev_new_indices[j]])
		A.lnkmatrices[tid][i, A.sortednodesperthread[tid, j]] += v
		return
	end
	
	if updatentryCSC2!(A.cscmatrix, i, j, v)
	else
		_, tid = last_nz(A.old_noderegions[:, A.rev_new_indices[j]])
		A.lnkmatrices[tid][i, A.sortednodesperthread[tid, j]] += v
	end
end

#---------------------------------

"""
$(SIGNATURES)
`updateindex!(ext::ExtendableSparseMatrixParallel{Tv, Ti}, op, v, i, j) where {Tv, Ti <: Integer`

Update element of the matrix  with operation `op`.
Use this method if the 'thread of the element' is not known, otherwise use `updateindex!(ext, op, v, i, j, tid)`. 
"""
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

"""
$(SIGNATURES)
`updateindex!(ext::ExtendableSparseMatrixParallel{Tv, Ti}, op, v, i, j, tid) where {Tv, Ti <: Integer`

Update element of the matrix  with operation `op`.
Use this method if the 'thread of the element' is known, otherwise use `updateindex!(ext, op, v, i, j)`. 
"""
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

"""
$(SIGNATURES)
`rawupdateindex!(ext::ExtendableSparseMatrixParallel{Tv, Ti}, op, v, i, j) where {Tv, Ti <: Integer}`

Like [`updateindex!`](@ref) but without checking if v is zero.
Use this method if the 'thread of the element' is not known.
"""
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

"""
$(SIGNATURES)
`rawupdateindex!(ext::ExtendableSparseMatrixParallel{Tv, Ti}, op, v, i, j, tid) where {Tv, Ti <: Integer}`

Like [`updateindex!`](@ref) but without checking if v is zero.
Use this method if the 'thread of the element' is known
"""
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

"""
$(SIGNATURES)
``Base.getindex(ext::ExtendableSparseMatrixParallel{Tv, Ti}, i::Integer, j::Integer) where {Tv, Ti <: Integer`

Find index in CSC matrix and return value, if it exists.
Otherwise, return value from extension.
"""
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

"""
$(SIGNATURES)
`Base.setindex!(ext::ExtendableSparseMatrixParallel{Tv, Ti}, v::Union{Number,AbstractVecOrMat}, i::Integer, j::Integer) where {Tv, Ti}`

Find index in CSC matrix and set value if it exists. Otherwise,
set index in extension if `v` is nonzero.
"""
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

"""
$(SIGNATURES)

Reset matrix, such that CSC and LNK have no non-zero entries.
"""
function reset!(A::ExtendableSparseMatrixParallel{Tv, Ti}) where {Tv, Ti <: Integer}
	A.cscmatrix = spzeros(Tv, Ti, A.n, A.m)
	A.lnkmatrices = [SuperSparseMatrixLNK{Tv, Ti}(A.n, A.nnts[tid]) for tid=1:A.nt]
end

"""
$(SIGNATURES)

Compute number of non-zero elements, after flush.
"""
function nnz_flush(ext::ExtendableSparseMatrixParallel)
    flush!(ext)
    return nnz(ext.cscmatrix)
end

"""
$(SIGNATURES)

Compute number of non-zero elements, without flush.
"""
function nnz_noflush(ext::ExtendableSparseMatrixParallel)
    return nnz(ext.cscmatrix), sum([ext.lnkmatrices[i].nnz for i=1:ext.nt])
end
	
function matrixindextype(A::ExtendableSparseMatrixParallel{Tv, Ti}) where {Tv, Ti <: Integer}
	Ti
end

function matrixvaluetype(A::ExtendableSparseMatrixParallel{Tv, Ti}) where {Tv, Ti <: Integer}
	Tv
end


"""
$(SIGNATURES)

Show matrix, without flushing
"""
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

"""
$(SIGNATURES)

Find out if i,j is non-zero entry in CSC, if yes, update entry with += v and return `true`, if not return `false`
"""
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



import LinearAlgebra.mul!

"""
$(SIGNATURES)
```function LinearAlgebra.mul!(y, A, x)```

This overwrites the mul! function for A::ExtendableSparseMatrixParallel
"""
function LinearAlgebra.mul!(y::AbstractVector{Tv}, A::ExtendableSparseMatrixParallel{Tv, Ti}, x::AbstractVector{Tv}) where {Tv, Ti<:Integer}
    #@info "my matvec"
    _, nnzLNK = nnz_noflush(A)
    @assert nnzLNK == 0
    #mul!(y, A.cscmatrix, x)
    matvec!(y, A, x)
end


"""
$(SIGNATURES)
```function matvec!(y, A, x)```

y <- A*x, where y and x are vectors and A is an ExtendableSparseMatrixParallel
this computation is done in parallel, it has the same result as y = A.cscmatrix*x
"""
function matvec!(y::AbstractVector{Tv}, A::ExtendableSparseMatrixParallel{Tv,Ti}, x::AbstractVector{Tv}) where {Tv, Ti<:Integer}
    nt = A.nt
    depth = A.depth
    colptr = A.cscmatrix.colptr
    nzv = A.cscmatrix.nzval
    rv  = A.cscmatrix.rowval

    LinearAlgebra._rmul_or_fill!(y, 0.0)
    
    for level=1:depth
        @threads for tid::Int64=1:nt
            for col::Int64=A.start[(level-1)*nt+tid]:A.start[(level-1)*nt+tid+1]-1
                for row::Int64=colptr[col]:colptr[col+1]-1 # in nzrange(A, col)
                    y[rv[row]] += nzv[row]*x[col]
                end
            end
        end
    end



    @threads for tid=1:1
        for col::Int64=A.start[depth*nt+1]:A.start[depth*nt+2]-1
            for row::Int64=colptr[col]:colptr[col+1]-1 #nzrange(A, col)
                y[rv[row]] += nzv[row]*x[col]
            end
        end
    end
    
    y
end
