"""
`function preparatory_multi_ps_less_reverse(nm, nt, depth)`

`nm` is the number of nodes in each dimension (Examples: 2d: nm = (100,100) -> 100 x 100 grid, 3d: nm = (50,50,50) -> 50 x 50 x 50 grid).
`nt` is the number of threads.
`depth` is the number of partition layers, for depth=1, there are nt parts and 1 separator, for depth=2, the separator is partitioned again, leading to 2*nt+1 submatrices...
To assemble the system matrix parallely, things such as `cellsforpart` (= which thread takes which cells) need to be computed in advance. This is done here.
"""
function preparatory_multi_ps_less_reverse(nm, nt, depth, Ti; sequential=false, x0=0.0, x1=1.0)
	grid = getgrid(nm; x0, x1)
	
	if sequential
		(allcells, start, cellparts) = grid_to_graph_ps_multi!(grid, nt, depth)#)
	else
		(allcells, start, cellparts) = grid_to_graph_ps_multi_par!(grid, nt, depth)
	end

	(nnts, s, onr, gi, gc, ni, rni, starts) = get_nnnts_and_sortednodesperthread_and_noderegs_from_cellregs_ps_less_reverse_nopush(
		cellparts, allcells, start, num_nodes(grid), Ti, nt
	)
	
	cfp = bettercellsforpart(cellparts, depth*nt+1)
	return grid, nnts, s, onr, cfp, gi, gc, ni, rni, starts, cellparts
end


"""
`function get_nnnts_and_sortednodesperthread_and_noderegs_from_cellregs_ps_less_reverse_nopush(cellregs, allcells, start, nn, Ti, nt)`

After the cellregions (partitioning of the grid) of the grid have been computed, other things have to be computed, such as `sortednodesperthread` a depth+1 x num_nodes matrix, here `sortednodesperthreads[i,j]` is the point at which the j-th node appears in the i-th level matrix in the corresponding submatrix.
`cellregs` contains the partiton for each cell.
Furthermore, `nnts` (number of nodes of the threads) is computed, which contain for each thread the number of nodes that are contained in the cells of that thread.
`allcells` and `start` together behave like the rowval and colptr arrays of a CSC matrix, such that `allcells[start[j]:start[j+1]-1]` are all cells that contain the j-th node.
`nn` is the number of nodes in the grid.
`Ti` is the type (Int64,...) of the elements in the created arrays.
`nt` is the number of threads.
"""
function get_nnnts_and_sortednodesperthread_and_noderegs_from_cellregs_ps_less_reverse_nopush(cellregs, allcells, start, nn, Ti, nt)
		
	num_matrices = maximum(cellregs)
	depth = Int(floor((num_matrices-1)/nt))

	#loop over each node, get the cellregion of the cell (the one not in the separator) write the position of that node inside the cellregions sorted ranking into a long vector
	#nnts = [zeros(Ti, nt+1) for i=1:depth+1]
	nnts = zeros(Ti, nt)
	#noderegs_max_tmp = 0
	old_noderegions = zeros(Ti, (depth+1, nn))
	
	# Count nodes per thread:
	tmp = zeros(depth+1)
	for j=1:nn
		cells = @view allcells[start[j]:start[j+1]-1]
		sortedcellregs = unique(sort(cellregs[cells]))
		#tmp = []
		tmpctr = 1
		for cr in sortedcellregs
			crmod = (cr-1)%nt+1
			level = Int(ceil(cr/nt))
			#nnts[crmod] += 1
			old_noderegions[level,j] = crmod
			if !(crmod in tmp[1:tmpctr-1])
				nnts[crmod] += 1
				#sortednodesperthread[crmod,j] = nnts[crmod] #nnts[i][cr]
				#push!(tmp, crmod)
				tmp[tmpctr] = crmod
				tmpctr += 1
			end
		end
	end
	
	# Reorder inidices to receive a block structure:
	# Taking the original matrix [a_ij] and mapping each i and j to new_indices[i] and new_indices[j], gives a block structure
	# the reverse is also defined rev_new_indices[new_indices[k]] = k
	# From now on we will only use this new ordering
	counter_for_reorder = zeros(Ti, depth*nt+1)
	for j=1:nn
		level, reg                                 = last_nz(old_noderegions[:, j])
		counter_for_reorder[(level-1)*nt + reg]    += 1 #(reg-1)*depth + level] += 1
	end
	
	starts               = vcat([0], cumsum(counter_for_reorder))
	counter_for_reorder2 = zeros(Ti, depth*nt+1)
	new_indices          = Vector{Ti}(undef, nn)
	rev_new_indices      = Vector{Ti}(undef, nn)
	origin               = Vector{Ti}(undef, nn)
	for j=1:nn
		level, reg                                  = last_nz(old_noderegions[:, j])
		counter_for_reorder2[(level-1)*nt + reg]    += 1
		origin[j]                                   = reg
		new_indices[j]                              = starts[(level-1)*nt + reg]+counter_for_reorder2[(level-1)*nt + reg]
		rev_new_indices[new_indices[j]]             = j
	end
	starts .+= 1
	
	# Build sortednodesperthread and globalindices array:
	# They are inverses of each other: globalindices[tid][sortednodeperthread[tid][j]] = j
	# Note that j has to be a `new index`
	
	sortednodesperthread = zeros(Ti, (nt, nn)) #vvcons(Ti, nnts)
	globalindices = vvcons(Ti, nnts)
	gictrs = zeros(Ti, nt)

	for nj=1:nn
		oj = rev_new_indices[nj]
		cells = @view allcells[start[oj]:start[oj+1]-1]
		sortedcellregs = unique(sort(cellregs[cells]))
		#tmp = []
		tmpctr = 1
		for cr in sortedcellregs
			crmod = (cr-1)%nt+1
			level = Int(ceil(cr/nt))
			if !(crmod in tmp[1:tmpctr-1])
				gictrs[crmod] += 1 # , level] += 1
				sortednodesperthread[crmod,nj] = gictrs[crmod]
				globalindices[crmod][gictrs[crmod]] = nj
				#push!(tmp, crmod)
				tmp[tmpctr] = crmod
				tmpctr += 1
			end
		end
	end
	
	nnts, sortednodesperthread, old_noderegions, globalindices, gictrs, new_indices, rev_new_indices, starts
end








"""
`function separate!(cellregs, nc, ACSC, nt, level0, ctr_sepanodes)`

This function partitons the separator, which is done if `depth`>1 (see `grid_to_graph_ps_multi!` and/or `preparatory_multi_ps`).
`cellregs` contains the regions/partitions/colors of each cell. 
`nc` is the number of cells in the grid.
`ACSC` is the adjacency matrix of the graph of the (separator-) grid (vertex in graph is cell in grid, edge in graph means two cells share a node) stored as a CSC. 
`nt` is the number of threads.
`level0` is the separator-partitoning level, if the (first) separator is partitioned, level0 = 1, in the next iteration, level0 = 2...
`preparatory_multi_ps` is the number of separator-cells.
"""
function separate!(cellregs, nc, ACSC, nt, level0, ctr_sepanodes)
	sepanodes = findall(x->x==nt+1, cellregs)

	indptr = collect(1:nc+1)
	indices = zeros(Int64, nc)
	rowval = zeros(Int64, nc)

	indptrT = collect(1:ctr_sepanodes+1)
	indicesT = zeros(Int64, ctr_sepanodes)
	rowvalT = zeros(Int64, ctr_sepanodes)

	for (i,j) in enumerate(sepanodes)
		indices[j] = i
		indicesT[i] = j
		rowval[j]  = 1
		rowvalT[i] = 1
	end

	R = SparseMatrixCSC(ctr_sepanodes, nc, indptr, indices, rowval)
	RT = SparseMatrixCSC(nc, ctr_sepanodes, indptrT, indicesT, rowvalT)
	prod = ACSC*dropzeros(RT)
	RART = dropzeros(R)*ACSC*dropzeros(RT)
	
	partition2 = Metis.partition(RART, nt)
	cellregs2 = copy(partition2)

	ctr_sepanodes = 0
	for (i,j) in enumerate(sepanodes)
		rows = RART.rowval[RART.colptr[i]:(RART.colptr[i+1]-1)]
		cellregs[j] = level0*nt + cellregs2[i]
		if minimum(partition2[rows]) != maximum(partition2[rows])
			cellregs[j] = (level0+1)*nt+1
			ctr_sepanodes += 1
		end
	end

	RART, ctr_sepanodes
end



"""
`function grid_to_graph_ps_multi!(grid, nt, depth)`

The function assigns colors/partitons to each cell in the `grid`. First, the grid is partitoned into `nt` partitions. If `depth` > 1, the separator is partitioned again...
`grid` is a simplexgrid. 
`nt` is the number of threads.
`depth` is the number of partition layers, for depth=1, there are nt parts and 1 separator, for depth=2, the separator is partitioned again, leading to 2*nt+1 submatrices...
"""
function grid_to_graph_ps_multi!(grid, nt, depth)
	A = SparseMatrixLNK{Int64, Int64}(num_cells(grid), num_cells(grid))
	number_cells_per_node = zeros(Int64, num_nodes(grid))
	for j=1:num_cells(grid)
		for node_id in grid[CellNodes][:,j]
			number_cells_per_node[node_id] += 1
		end
	end
	allcells = zeros(Int64, sum(number_cells_per_node))
	start = ones(Int64, num_nodes(grid)+1)
	start[2:end] += cumsum(number_cells_per_node)
	number_cells_per_node .= 0
	for j=1:num_cells(grid)
		for node_id in grid[CellNodes][:,j]
			allcells[start[node_id] + number_cells_per_node[node_id]] = j
			number_cells_per_node[node_id] += 1
		end
	end

	for j=1:num_nodes(grid)
		cells = @view allcells[start[j]:start[j+1]-1]
		for (i,id1) in enumerate(cells)
			for id2 in cells[i+1:end]
				A[id1,id2] = 1
				A[id2,id1] = 1
			end
		end	
	end

	ACSC = SparseArrays.SparseMatrixCSC(A)
	
	partition = Metis.partition(ACSC, nt)
	cellregs  = copy(partition)
	
	ctr_sepanodes = 0
	for j=1:num_cells(grid)
		rows = ACSC.rowval[ACSC.colptr[j]:(ACSC.colptr[j+1]-1)]
		if minimum(partition[rows]) != maximum(partition[rows])
			cellregs[j] = nt+1
			ctr_sepanodes += 1
		end
	end
	RART = ACSC
	for level=1:depth-1
		RART, ctr_sepanodes = separate!(cellregs, num_cells(grid), RART, nt, level, ctr_sepanodes)
	end

			
	return allcells, start, cellregs
end



function grid_to_graph_ps_multi_par!(grid, nt, depth)
	time = zeros(12)
	As = [ExtendableSparseMatrix{Int64, Int64}(num_cells(grid), num_cells(grid)) for tid=1:nt]
	number_cells_per_node = zeros(Int64, num_nodes(grid))
	
	cn = grid[CellNodes]
	
	for j=1:num_cells(grid)
		tmp = view(cn, :, j)
		for node_id in tmp
			number_cells_per_node[node_id] += 1
		end
	end
		
	
	allcells = zeros(Int64, sum(number_cells_per_node))
	start = ones(Int64, num_nodes(grid)+1)
	start[2:end] += cumsum(number_cells_per_node)
	number_cells_per_node .= 0
	
	for j=1:num_cells(grid)
		tmp = view(cn, :, j)
		for node_id in tmp
			allcells[start[node_id] + number_cells_per_node[node_id]] = j
			number_cells_per_node[node_id] += 1
		end
	end

	node_range = get_starts(num_nodes(grid), nt)
	Threads.@threads for tid=1:nt
		for j in node_range[tid]:node_range[tid+1]-1
			cells = @view allcells[start[j]:start[j+1]-1]
			l = length(cells)
			for (i,id1) in enumerate(cells)
				ce = view(cells, i+1:l)
				for id2 in ce
					As[tid][id1,id2] = 1
					As[tid][id2,id1] = 1
				end
			end	
		end
		ExtendableSparse.flush!(As[tid])
	end
	
	ACSC = add_all_par!(As).cscmatrix
		
	#SparseArrays.SparseMatrixCSC(A))
	
	
	partition = Metis.partition(ACSC, nt)
	cellregs  = copy(partition)
	
	ctr_sepanodes_a = zeros(Int64, nt)
	
	cell_range = get_starts(num_cells(grid), nt)
	Threads.@threads :static for tid=1:nt
		for j in cell_range[tid]:cell_range[tid+1]-1
			rows = @view ACSC.rowval[ACSC.colptr[j]:(ACSC.colptr[j+1]-1)]
			if minimum(partition[rows]) != maximum(partition[rows])
				cellregs[j] = nt+1
				ctr_sepanodes_a[tid] += 1
			end
		end
	end
	
	ctr_sepanodes = sum(ctr_sepanodes_a)
			
	#=
	time[10] = @elapsed for j=1:num_cells(grid)
		rows = ACSC.rowval[ACSC.colptr[j]:(ACSC.colptr[j+1]-1)]
		if minimum(partition[rows]) != maximum(partition[rows])
			cellregs[j] = nt+1
			ctr_sepanodes += 1
		end
	end
	=#
	RART = ACSC
	for level=1:depth-1
		RART, ctr_sepanodes = separate!(cellregs, num_cells(grid), RART, nt, level, ctr_sepanodes)
	end

			
	return allcells, start, cellregs
end


function add_all_par!(As)
	nt = length(As)
	depth = Int(floor(log2(nt)))
	ende = nt
	for level=1:depth
		
		@threads :static for tid=1:2^(depth-level)
			#@info "$level, $tid"
			start = tid+2^(depth-level)
			while start <= ende
				As[tid] += As[start]
				start += 2^(depth-level)
			end
		end
		ende = 2^(depth-level)
	end
	As[1]

end


"""
`function vvcons(Ti, lengths)`

`lengths` is a vector of integers.
The function creates a vector of zero vectors of type `Ti` of length `lengths[i]`.
"""
function vvcons(Ti, lengths)
	x::Vector{Vector{Ti}} = [zeros(Ti, i) for i in lengths]
	return x
end


"""
`function bettercellsforpart(xx, upper)`

`xx` are the CellRegions (i.e. the color/partition of each cell).
`upper` is the number of partitions (upper=depth*nt+1).
The function returns a vector e.g. [v1, v2, v3, v4, v5].
The element v1 would be the list of cells that are in partition 1 etc.
The function is basically a faster findall.
"""
function bettercellsforpart(xx, upper)
	ctr = zeros(Int64, upper)
	for x in xx
		ctr[x] += 1
	end
	cfp = vvcons(Int64, ctr)
	ctr .= 1
	for (i,x) in enumerate(xx)
		cfp[x][ctr[x]] = i
		ctr[x] += 1
	end
	cfp
end

"""
`function getgrid(nm)`

Returns a simplexgrid with a given number of nodes in each dimension.
`nm` is the number of nodes in each dimension (Examples: 2d: nm = (100,100) -> 100 x 100 grid, 3d: nm = (50,50,50) -> 50 x 50 x 50 grid).
"""
function getgrid(nm; x0=0.0, x1=1.0)
	if length(nm) == 2
		n,m = nm
		xx = collect(LinRange(x0, x1, n))
		yy = collect(LinRange(x0, x1, m))
		grid = simplexgrid(xx, yy)
	else 
		n,m,l = nm
		xx = collect(LinRange(x0, x1, n))
		yy = collect(LinRange(x0, x1, m))
		zz = collect(LinRange(x0, x1, l))
		grid = simplexgrid(xx, yy, zz)
	end
	grid
end

function get_starts(n, nt)
	ret = ones(Int64, nt+1)
	ret[end] = n+1
	for i=nt:-1:2
		ret[i] = ret[i+1] - Int(round(ret[i+1]/i)) #Int(round(n/nt))-1
	end 
	ret
end

function last_nz(x)
	n = length(x)
	for j=n:-1:1
		if x[j] != 0
			return (j, x[j])
		end
	end
end

