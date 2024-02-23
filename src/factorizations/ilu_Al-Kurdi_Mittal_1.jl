module ILUAM
using LinearAlgebra, SparseArrays

#import LinearAlgebra.ldiv!, LinearAlgebra.\, SparseArrays.nnz 

@info "ILUAM"

mutable struct ILUAMPrecon{T,N}

	diag::AbstractVector
    nzval::AbstractVector
	A::AbstractMatrix
	
end

function ILUAMPrecon(A::SparseMatrixCSC{T,N}, b_type=T) where {T,N<:Integer}
    @info "ILUAMPrecon"
	n      = A.n # number of columns
	nzval  = copy(A.nzval)
	diag   = Vector{N}(undef, n)
	
    ILUAMPrecon{T, N}(diag, copy(A.nzval), A)
end



function iluAM!(LU::ILUAMPrecon{T,N}, A::SparseMatrixCSC{T,N}) where {T,N<:Integer}
    @info "iluAM!"
	nzval  = LU.nzval
    diag   = LU.diag
    
    colptr = LU.A.colptr
	rowval = LU.A.rowval
	n      = A.n # number of columns
	point  = zeros(N, n) #Vector{N}(undef, n)
	
	t = zeros(5)

	# find diagonal entries
	t[1] = @elapsed for j=1:n
		for v=colptr[j]:colptr[j+1]-1
			if rowval[v] == j
				diag[j] = v
				break
			end
			#elseif rowval[v] 
		end
	end
	
	# compute L and U
	for j=1:n
		t[2] += @elapsed for v=colptr[j]:colptr[j+1]-1  ## start at colptr[j]+1 ??
			point[rowval[v]] = v
		end
		
		t[3] += @elapsed for v=colptr[j]:diag[j]-1
			i = rowval[v]
			#nzval[v] /= nzval[diag[i]]
			for w=diag[i]+1:colptr[i+1]-1
				k = point[rowval[w]]
				if k>0
					nzval[k] -= nzval[v]*nzval[w]
				end
			end
		end
		
		t[4] += @elapsed for v=diag[j]+1:colptr[j+1]-1
			nzval[v] /= nzval[diag[j]]
		end
		
		
		t[5] += @elapsed for v=colptr[j]:colptr[j+1]-1
			point[rowval[v]] = zero(N)
		end
	end
	t
end


function iluAM(A::SparseMatrixCSC{Tv,Ti}) where {Tv, Ti <:Integer}
	@info "iluAM"
	nzval = copy(A.nzval)
	colptr = A.colptr
	rowval = A.rowval
	#nzval  = ILU.nzval
	n = A.n # number of columns
	point = zeros(Ti, n) #Vector{Ti}(undef, n)
	diag  = Vector{Ti}(undef, n)
	
	# find diagonal entries
	for j=1:n
		for v=colptr[j]:colptr[j+1]-1
			if rowval[v] == j
				diag[j] = v
				break
			end
			#elseif rowval[v] 
		end
	end
	
	# compute L and U
	for j=1:n
		for v=colptr[j]:colptr[j+1]-1  ## start at colptr[j]+1 ??
			point[rowval[v]] = v
		end
		
		for v=colptr[j]:diag[j]-1
			i = rowval[v]
			#nzval[v] /= nzval[diag[i]]
			for w=diag[i]+1:colptr[i+1]-1
				k = point[rowval[w]]
				if k>0
					nzval[k] -= nzval[v]*nzval[w]
				end
			end
		end
		
		for v=diag[j]+1:colptr[j+1]-1
			nzval[v] /= nzval[diag[j]]
		end
		
		
		for v=colptr[j]:colptr[j+1]-1
			point[rowval[v]] = zero(Ti)
		end
	end
	#nzval, diag
	ILUAMPrecon{Tv,Ti}(diag, nzval, A)
end

#function iluAM(A::SparseMatrixCSC{T,N}) where {T,N<:Integer}
#    t = zeros(6)
#	t[1] = @elapsed (LU = ILUAMPrecon(A::SparseMatrixCSC{T,N}))
#    t[2:6] = iluAM!(LU, A)
#	@info t
#    LU
#end


function forward_substitution!(y, ilu::ILUAMPrecon{T,N}, v) where {T,N<:Integer}
	n = ilu.A.n
	nzval = ilu.nzval
	colptr = ilu.A.colptr
	rowval = ilu.A.rowval
	diag = ilu.diag
	y .= 0
	@inbounds for j=1:n
		y[j] += v[j]
		for v=diag[j]+1:colptr[j+1]-1
			y[rowval[v]] -= nzval[v]*y[j]
		end
	end
	y
end


function backward_substitution!(x, ilu::ILUAMPrecon{T,N}, y) where {T,N<:Integer}
    n = ilu.A.n
	nzval = ilu.nzval
	colptr = ilu.A.colptr
	rowval = ilu.A.rowval
	diag = ilu.diag
	wrk = copy(y)
	@inbounds for j=n:-1:1
		x[j] = wrk[j] / nzval[diag[j]]		
		for i=colptr[j]:diag[j]-1
			wrk[rowval[i]] -= nzval[i]*x[j]
		end
	end
    x
end

function ldiv_new!(x, ilu, v)

	n = ilu.A.n
	y = Vector{Float64}(undef, n)
	y .= 0
	nzval = ilu.nzval
	colptr = ilu.A.colptr
	rowval = ilu.A.rowval
	diag = ilu.diag
	#forward
	@inbounds for j=1:n
		y[j] += v[j]
		for v=diag[j]+1:colptr[j+1]-1
			y[rowval[v]] -= nzval[v]*y[j]
		end
	end

	#backward
	wrk = copy(y)
	@inbounds for j=n:-1:1
		x[j] = wrk[j] / nzval[diag[j]]		
		for i=colptr[j]:diag[j]-1
			wrk[rowval[i]] -= nzval[i]*x[j]
		end
	end
    x
end

function ldiv!(x, ilu::ILUAMPrecon{T,N}, b) where {T,N<:Integer}
	#@info "AM ldiv1"
    y = copy(b)
    forward_substitution!(y, ilu, b)
    backward_substitution!(x, ilu, y)
    x
end

function ldiv!(ilu::ILUAMPrecon{T,N}, b) where {T,N<:Integer}
    @info "AM ldiv2"
	y = copy(b)
    forward_substitution!(y, ilu, b)
    backward_substitution!(b, ilu, y)
    b
end

function \(ilu::ILUAMPrecon{T,N}, b) where {T,N<:Integer}
    @info "AM bs "
	x = copy(b)
    ldiv!(x, ilu, b)
	x
end

function nnz(ilu::ILUAMPrecon{T,N}) where {T,N<:Integer}
    length(ilu.nzval)
end


end