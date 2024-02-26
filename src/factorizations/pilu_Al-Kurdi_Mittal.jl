#module PILUAM
#using Base.Threads
#using LinearAlgebra, SparseArrays

import LinearAlgebra.ldiv!, LinearAlgebra.\, SparseArrays.nnz 

@info "PILUAM"

mutable struct PILUAMPrecon{T,N}

	diag::AbstractVector
    nzval::AbstractVector
	A::AbstractMatrix
	start::AbstractVector
	nt::Integer
	depth::Integer
	
end

function use_vector_par(n, nt, Ti)
	point = [Vector{Ti}(undef, n) for tid=1:nt]
	@threads for tid=1:nt
		point[tid] = zeros(Ti, n)
	end
	point
end

function compute_lu!(nzval, point, j0, j1, tid, rowval, colptr, diag, Ti)
    for j=j0:j1-1
        for v=colptr[j]:colptr[j+1]-1
            point[tid][rowval[v]] = v
        end
        
        for v=colptr[j]:diag[j]-1
            i = rowval[v]
            for w=diag[i]+1:colptr[i+1]-1
                k = point[tid][rowval[w]]
                if k>0
                    nzval[k] -= nzval[v]*nzval[w]
                end
            end
        end
        
        for v=diag[j]+1:colptr[j+1]-1
            nzval[v] /= nzval[diag[j]]
        end
        
        for v=colptr[j]:colptr[j+1]-1
            point[tid][rowval[v]] = zero(Ti)
        end
    end
end

function piluAM(A::ExtendableSparseMatrixParallel{Tv,Ti}) where {Tv, Ti <:Integer}
	start = A.start
	nt = A.nt
	depth = A.depth
	
	colptr = A.cscmatrix.colptr
	rowval = A.cscmatrix.rowval
	nzval  = Vector{Tv}(undef, length(rowval)) #copy(A.nzval)
	n = A.cscmatrix.n # number of columns
	diag  = Vector{Ti}(undef, n)
	point = use_vector_par(n, A.nt, Int32)
	
	# find diagonal entries
	#
	@threads for tid=1:depth*nt+1
		for j=start[tid]:start[tid+1]-1
			for v=colptr[j]:colptr[j+1]-1
				nzval[v] = A.cscmatrix.nzval[v]
				if rowval[v] == j
					diag[j] = v
				end
				#elseif rowval[v] 
			end
		end
	end

	#=
	@info "piluAM"
	nzval = copy(A.cscmatrix.nzval)
	colptr = A.cscmatrix.colptr
	rowval = A.cscmatrix.rowval
	#nzval  = ILU.nzval
	n = A.n # number of columns
	diag  = Vector{Ti}(undef, n)
	start = A.start
    nt    = A.nt
    depth = A.depth
    point = use_vector_par(n, nt, Ti)

	# find diagonal entries
    @threads for tid=1:depth*nt+1
        for j=start[tid]:start[tid+1]-1
            for v=colptr[j]:colptr[j+1]-1
                if rowval[v] == j
                    diag[j] = v
                    break
                end
                #elseif rowval[v] 
            end
        end
    end
	
	# compute L and U
    for level=1:depth
        @threads for tid=1:nt
            compute_lu!(nzval, point, start[(level-1)*nt+tid], start[(level-1)*nt+tid+1], tid, rowval, colptr, diag, Ti)
        end
    end

    compute_lu!(nzval, point, start[depth*nt+1], start[depth*nt+2], 1, rowval, colptr, diag, Ti)
	=#

	for level=1:depth
		@threads for tid=1:nt
			for j=start[(level-1)*nt+tid]:start[(level-1)*nt+tid+1]-1
				for v=colptr[j]:colptr[j+1]-1
					point[tid][rowval[v]] = v
				end
				
				for v=colptr[j]:diag[j]-1
					i = rowval[v]
					for w=diag[i]+1:colptr[i+1]-1
						k = point[tid][rowval[w]]
						if k>0
							nzval[k] -= nzval[v]*nzval[w]
						end
					end
				end
				
				for v=diag[j]+1:colptr[j+1]-1
					nzval[v] /= nzval[diag[j]]
				end
				
				for v=colptr[j]:colptr[j+1]-1
					point[tid][rowval[v]] = zero(Ti)
				end
			end
		end
	end
	
	#point = zeros(Ti, n) #Vector{Ti}(undef, n)
	for j=start[depth*nt+1]:start[depth*nt+2]-1
		for v=colptr[j]:colptr[j+1]-1
			point[1][rowval[v]] = v
		end
		
		for v=colptr[j]:diag[j]-1
			i = rowval[v]
			for w=diag[i]+1:colptr[i+1]-1
				k = point[1][rowval[w]]
				if k>0
					nzval[k] -= nzval[v]*nzval[w]
				end
			end
		end
		
		for v=diag[j]+1:colptr[j+1]-1
			nzval[v] /= nzval[diag[j]]
		end
		
		for v=colptr[j]:colptr[j+1]-1
			point[1][rowval[v]] = zero(Ti)
		end
	end

	#nzval, diag
	PILUAMPrecon{Tv,Ti}(diag, nzval, A.cscmatrix, start, nt, depth)
end

function forward_subst_old!(y, v, nzval, diag, start, nt, depth, A)
	#@info "fwo"
	n = A.n
	colptr = A.colptr
	rowval = A.rowval
	
	y .= 0
	
	for level=1:depth
		@threads for tid=1:nt
			@inbounds for j=start[(level-1)*nt+tid]:start[(level-1)*nt+tid+1]-1
				y[j] += v[j]
				for v=diag[j]+1:colptr[j+1]-1
					y[rowval[v]] -= nzval[v]*y[j]
				end
			end
		end
	end
	
	@inbounds for j=start[depth*nt+1]:start[depth*nt+2]-1
		y[j] += v[j]
		for v=diag[j]+1:colptr[j+1]-1
			y[rowval[v]] -= nzval[v]*y[j]
		end
	end
	
end


function backward_subst_old!(x, y, nzval, diag, start, nt, depth, A)
	#@info "bwo"
	n = A.n
	colptr = A.colptr
	rowval = A.rowval
	#wrk = copy(y)
	
	
	@inbounds for j=start[depth*nt+2]-1:-1:start[depth*nt+1]
		x[j] = y[j] / nzval[diag[j]] 
		
		for i=colptr[j]:diag[j]-1
			y[rowval[i]] -= nzval[i]*x[j]
		end
		
	end
	
	for level=depth:-1:1
		@threads for tid=1:nt
			@inbounds for j=start[(level-1)*nt+tid+1]-1:-1:start[(level-1)*nt+tid]
				x[j] = y[j] / nzval[diag[j]] 
				for i=colptr[j]:diag[j]-1
					y[rowval[i]] -= nzval[i]*x[j]
				end
			end
		end
	end

end

function ldiv!(x, ILU::PILUAMPrecon, b)
	#@info "piluam ldiv 1"
	nzval = ILU.nzval
	diag  = ILU.diag
	A     = ILU.A
    start = ILU.start
    nt    = ILU.nt
    depth = ILU.depth
	y = copy(b)
	#forward_subst!(y, b, ILU)
	forward_subst_old!(y, b, nzval, diag, start, nt, depth, A)
	backward_subst_old!(x, y, nzval, diag, start, nt, depth, A)
	x
end

function ldiv!(ILU::PILUAMPrecon, b)
	#@info "piluam ldiv 2"
	nzval = ILU.nzval
	diag  = ILU.diag
	A     = ILU.A
    start = ILU.start
    nt    = ILU.nt
    depth = ILU.depth
	y = copy(b)
	#forward_subst!(y, b, ILU)
	forward_subst_old!(y, b, nzval, diag, start, nt, depth, A)
	backward_subst_old!(b, y, nzval, diag, start, nt, depth, A)
	b
end

function \(ilu::PILUAMPrecon{T,N}, b) where {T,N<:Integer}
    x = copy(b)
    ldiv!(x, ilu, b)
	x
end

function nnz(ilu::PILUAMPrecon{T,N}) where {T,N<:Integer}
    length(ilu.nzval)
end

#end