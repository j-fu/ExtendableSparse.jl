#module PILUAM
#using Base.Threads
#using LinearAlgebra, SparseArrays

import LinearAlgebra.ldiv!, LinearAlgebra.\, SparseArrays.nnz 

@info "PILUAM"

mutable struct PILUAMPrecon{T,N}

	diag::AbstractVector
    nzval::AbstractVector
	A::AbstractMatrix
	
end

function iluAM!(ILU::PILUAMPrecon{Tv,Ti}, A::ExtendableSparseMatrixParallel{Tv, Ti}) where {Tv, Ti <:Integer}
	@info "filuAM!"
    diag = ILU.diag
	nzval = ILU.nzval

	nzval = copy(A.cscmatrix.nzval)
	diag  = Vector{Ti}(undef, n)
	ILU.A = A
	colptr = A.cscmatrix.colptr
	rowval = A.cscmatrix.rowval
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


function piluAM(A::ExtendableSparseMatrixParallel{Tv,Ti}) where {Tv, Ti <:Integer}
	@info "filuAM, $(A[1,1])"
	nzval = copy(A.cscmatrix.nzval)
	colptr = A.cscmatrix.colptr
	rowval = A.cscmatrix.rowval
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
	PILUAMPrecon{Tv,Ti}(diag, nzval, A)
end



function ldiv!(x, ILU::PILUAMPrecon, b)
	#@info "iluam ldiv 1"
	nzval = ILU.nzval
	diag  = ILU.diag
	A     = ILU.A.cscmatrix
	y = copy(b)
	#forward_subst!(y, b, ILU)
	forward_subst_old!(y, b, nzval, diag, A)
	backward_subst_old!(x, y, nzval, diag, A)
	@info "FILUAM:", b[1], y[1], x[1], maximum(abs.(b-A*x)) 
    #, maximum(abs.(b-A*x)), b[1], x[1], y[1]
	x
end


function ldiv!(ILU::PILUAMPrecon, b)
	#@info "iluam ldiv 2"
	nzval = ILU.nzval
	diag  = ILU.diag
	A     = ILU.A.cscmatrix
	y = copy(b)
	#forward_subst!(y, b, ILU)
	forward_subst_old!(y, b, nzval, diag, A)
	backward_subst_old!(b, y, nzval, diag, A)
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