mutable struct ParallelILU0Preconditioner{Tv, Ti} <: AbstractPreconditioner{Tv,Ti}
    A::ExtendableSparseMatrix{Tv,Ti}
    xdiag::Array{Tv,1}
    idiag::Array{Ti,1}
    phash::UInt64

    coloring::Array{Array{Ti,1},1}
    coloring_index::Array{Array{Ti,1},1}
    coloring_index_reverse::Array{Array{Ti,1},1}

    function ParallelILU0Preconditioner{Tv,Ti}() where {Tv,Ti}
        p=new()
        p.phash=0
        p
    end
end

"""
```
ParallelILU0Preconditioner(;valuetype=Float64,indextype=Int64)
ParallelILU0Preconditioner(matrix)
```

Parallel ILU preconditioner with zero fill-in.
"""
ParallelILU0Preconditioner(;valuetype::Type=Float64, indextype::Type=Int64)=ParallelILU0Preconditioner{valuetype,indextype}()


function update!(precon::ParallelILU0Preconditioner{Tv,Ti}) where {Tv,Ti}
    flush!(precon.A)

    # Get coloring and reorder matrix
    precon.coloring=graphcol(precon.A.cscmatrix)
    precon.coloring_index, precon.coloring_index_reverse=coloringindex(precon.coloring)
    precon.A=ExtendableSparseMatrix(reordermatrix(precon.A.cscmatrix, precon.coloring))

    cscmatrix=precon.A.cscmatrix
    colptr=cscmatrix.colptr
    rowval=cscmatrix.rowval
    nzval=cscmatrix.nzval
    n=cscmatrix.n

    if precon.phash==0
        n=size(precon.A,1)
        precon.xdiag=Array{Tv,1}(undef,n)
        precon.idiag=Array{Ti,1}(undef,n)
    end

    xdiag=precon.xdiag
    idiag=precon.idiag

    
    # Find main diagonal index and
    # copy main diagonal values
    if precon.phash != precon.A.phash
        @inbounds for j=1:n
            @inbounds for k=colptr[j]:colptr[j+1]-1
                i=rowval[k]
                if i==j
                    idiag[j]=k
                    break
                end
            end
        end
        precon.phash=precon.A.phash
    end

    @inbounds for j=1:n
        xdiag[j]=one(Tv)/nzval[idiag[j]]
        @inbounds for k=idiag[j]+1:colptr[j+1]-1
            i=rowval[k]
            for l=colptr[i]:colptr[i+1]-1
                if rowval[l]==j
                    xdiag[i]-=nzval[l]*xdiag[j]*nzval[k]
                    break
                end
            end
        end
    end
    precon
end


function  LinearAlgebra.ldiv!(u::AbstractArray{T,1}, precon::ParallelILU0Preconditioner, v::AbstractArray{T,1}) where T
    cscmatrix=precon.A.cscmatrix
    colptr=cscmatrix.colptr
    rowval=cscmatrix.rowval
    n=cscmatrix.n
    nzval=cscmatrix.nzval
    xdiag=precon.xdiag
    idiag=precon.idiag

    coloring = precon.coloring
    coloring_index = precon.coloring_index
    coloring_index_reverse = precon.coloring_index_reverse
    
    @inbounds for indset in coloring_index
	    @inbounds Threads.@threads for j in indset
	        x=zero(T)
	        @inbounds for k=colptr[j]:idiag[j]-1
	            x+=nzval[k]*u[rowval[k]]
	        end
	        u[j]=xdiag[j]*(v[j]-x)
	    end
	end
    
    @inbounds for indset in coloring_index_reverse
    	@inbounds Threads.@threads for j in indset
	        x=zero(T)
	        @inbounds for k=idiag[j]+1:colptr[j+1]-1
	            x+=u[rowval[k]]*nzval[k]
	        end
	        u[j]-=x*xdiag[j]
	    end
	end
end


function LinearAlgebra.ldiv!(precon::ParallelILU0Preconditioner, v::AbstractArray{T,1} where T)
    ldiv!(v, precon, v)
end


# Returns an independent set of the graph of a matrix
# Reference: https://research.nvidia.com/sites/default/files/pubs/2015-05_Parallel-Graph-Coloring/nvr-2015-001.pdf
function indset(A::SparseMatrixCSC{Tv,Ti}, W::StridedVector{Ti}) where {Tv,Ti}
	# Random numbers for all vertices
	lenW = length(W)
	# r = sample(1:lenW, lenW, replace = false)
	r = rand(lenW)
	@inbounds for i = 1:lenW
		if W[i] == 0 
			r[i] = 0
		end
	end
	# Empty independent set
	S = zeros(Int, lenW)
	# Get independent set by comparing random number of vertex with the random 
	# numbers of all neighbor vertices
	@inbounds Threads.@threads for i in 1:lenW
		if W[i] != 0
			j = A.rowval[A.colptr[i]:A.colptr[i+1]-1]
			if all(x->x==1, r[i] .>= r[j])
				S[i] = W[i]
			end
		end
	end
	# Remove zero entries and return independent set
	return filter!(x->x≠0, S)
end

# Returns coloring of the graph of a matrix
# Reference: https://research.nvidia.com/sites/default/files/pubs/2015-05_Parallel-Graph-Coloring/nvr-2015-001.pdf
function graphcol(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
	# Empty list for coloring
	C = []
	# Array of vertices
	W = [1:size(A)[1];]
	# Get all independent sets of the graph of the matrix
	while any(W .!= 0)
		# Get independent set
		S = indset(A + transpose(A), W)
		push!(C, S)
		# Remove entries in S from W
		@inbounds for s in S
			W[s] = 0
		end
	end
	# Return coloring
	return C 
end


# Reorders a sparse matrix with provided coloring
function reordermatrix(A::SparseMatrixCSC{Tv,Ti}, coloring::Array{Array{Int64,1},1}) where {Tv,Ti}
	c = collect(Iterators.flatten(coloring))
	return A[c,:][:,c]
end

# Reorders a linear system with provided coloring
function reorderlinsys(A::SparseMatrixCSC{Tv,Ti}, b::StridedVector{Tv}, coloring::Array{Array{Int64,1},1}) where {Tv,Ti}
	c = collect(Iterators.flatten(coloring))
	return A[c,:][:,c], b[c]
end


# Returns an array with the same structure of the input coloring and ordered 
# entries 1:length(coloring) and an array with the structure of 
# reverse(coloring) and ordered entries length(coloring):-1:1 
function coloringindex(coloring::Array{Array{Int64,1},1})
    # First array
	c = deepcopy(coloring)
	cnt = 1
	@inbounds for i in 1:length(c)
		@inbounds for j in 1:length(c[i])
			c[i][j] = cnt
			cnt += 1
		end
	end
    # Second array
	cc = deepcopy(reverse(coloring))
	@inbounds for i in 1:length(cc)
		@inbounds for j in 1:length(cc[i])
			cnt -= 1
			cc[i][j] = cnt
		end
	end
    # Return both
	return c, cc
end
