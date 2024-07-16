#module ILUAM
#using LinearAlgebra, SparseArrays

import LinearAlgebra.ldiv!, LinearAlgebra.\, SparseArrays.nnz 

#@info "ILUAM"

mutable struct ILUAMPrecon{T,N}

	diag::AbstractVector
    nzval::AbstractVector
	A::AbstractMatrix
	
end


function iluAM!(ILU::ILUAMPrecon{Tv,Ti}, A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti <:Integer}
	diag = ILU.diag
	nzval = ILU.nzval

	nzval = copy(A.nzval)
	diag  = Vector{Ti}(undef, n)
	ILU.A = A
	colptr = A.colptr
	rowval = A.rowval
	n = A.n
	point = zeros(Ti, n)

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

end

function iluAM(A::SparseMatrixCSC{Tv,Ti}) where {Tv, Ti <:Integer}
	#@info "iluAM"
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

	#@info diag[1:20]'
	#@info diag[end-20:end]'
	
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

function forward_subst_old!(y, v, nzval, diag, A)
	#@info "fso, $(sum(nzval)), $(sum(nzval.^2)), $(sum(diag)), $(A[1,1])"
	n      = A.n
	colptr = A.colptr
	rowval = A.rowval
	
	for i in eachindex(y)
        y[i] = zero(Float64)
    end

	#y .= 0
	@inbounds for j=1:n
		y[j] += v[j]
		for v=diag[j]+1:colptr[j+1]-1
			y[rowval[v]] -= nzval[v]*y[j]
		end
	end
	y
end


function backward_subst_old!(x, y, nzval, diag, A)
	#@info "bso, $(sum(nzval)), $(sum(nzval.^2)), $(sum(diag)), $(A[1,1])"
	n      = A.n
	colptr = A.colptr
	rowval = A.rowval
	@inbounds for j=n:-1:1
		x[j] = y[j] / nzval[diag[j]] 
		
		for i=colptr[j]:diag[j]-1
			y[rowval[i]] -= nzval[i]*x[j]
		end
		
	end
	x
end


function ldiv!(x, ILU::ILUAMPrecon, b)
	#t = @elapsed begin
	#@info "iluam ldiv 1"
	nzval = ILU.nzval
	diag  = ILU.diag
	A     = ILU.A
	y = copy(b)
	#forward_subst!(y, b, ILU)
	forward_subst_old!(y, b, nzval, diag, A)
	backward_subst_old!(x, y, nzval, diag, A)
	#@info "ILUAM:", b[1], y[1], x[1], maximum(abs.(b-A*x)), nnz(A) #, A[10,10]
	#, b[1], x[1], y[1]#maximum(abs.(b)), maximum(abs.(x))
	#end
	#println("$t") #@info t
	x
end

function ldiv!(ILU::ILUAMPrecon, b)
	#@info "iluam ldiv 2"
	nzval = ILU.nzval
	diag  = ILU.diag
	A     = ILU.A
	y = copy(b)
	#forward_subst!(y, b, ILU)
	forward_subst_old!(y, b, nzval, diag, A)
	backward_subst_old!(b, y, nzval, diag, A)
	b
end

function \(ilu::ILUAMPrecon{T,N}, b) where {T,N<:Integer}
    x = copy(b)
    ldiv!(x, ilu, b)
	x
end

function nnz(ilu::ILUAMPrecon{T,N}) where {T,N<:Integer}
    length(ilu.nzval)
end

#=
function forward_subst!(y, v, ilu) #::ILUAMPrecon{T,N}) where {T,N<:Integer}
	@info "fw"
	n = ilu.A.n
	nzval  = ilu.nzval
	diag   = ilu.diag
	colptr = ilu.A.colptr
	rowval = ilu.A.rowval
	
	for i in eachindex(y)
        y[i] = zero(Float64)
    end

	#y .= 0
	@inbounds for j=1:n
		y[j] += v[j]
		for v=diag[j]+1:colptr[j+1]-1
			y[rowval[v]] -= nzval[v]*y[j]
		end
	end
	y
end

function backward_subst!(x, y, ilu) #::ILUAMPrecon{T,N}) where {T,N<:Integer}
    @info "bw"
	n = ilu.A.n
	nzval = ilu.nzval
	diag = ilu.diag
	colptr = ilu.A.colptr
	rowval = ilu.A.rowval
	#wrk = copy(y)
	@inbounds for j=n:-1:1
		x[j] = y[j] / nzval[diag[j]] 
		
		for i=colptr[j]:diag[j]-1
			y[rowval[i]] -= nzval[i]*x[j]
		end
		
	end
	x
end

function iluam_subst(ILU::ILUAMPrecon, b)
	y = copy(b)
	forward_subst!(y, b, ILU)
	z = copy(b)
	backward_subst!(z, y, ILU)
	z
end



function iluam_subst_old(ILU::ILUAMPrecon, b)
	nzval = ILU.nzval
	diag  = ILU.diag
	A     = ILU.A
	y = copy(b)
	#forward_subst!(y, b, ILU)
	forward_subst_old!(y, b, nzval, diag, A)
	z = copy(b)
	backward_subst_old!(z, y, nzval, diag, A)
	#backward_subst!(z, y, ILU)
	z
end
=#



#end