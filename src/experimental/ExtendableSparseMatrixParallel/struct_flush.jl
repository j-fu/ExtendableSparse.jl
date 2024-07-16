function flush!(A::ExtendableSparseMatrixParallel; do_dense=false, keep_zeros=true)
	_, nnzLNK = nnz_noflush(A)

	if nnzLNK == 0
		return
	end

	if !do_dense
		A.cscmatrix = A.cscmatrix+sparse_flush!(A; keep_zeros)
	
	else
		if keep_zeros
			A.cscmatrix = dense_flush_keepzeros!(A.lnkmatrices, A.old_noderegions, A.sortednodesperthread, A.nt, A.rev_new_indices)
		else
			A.cscmatrix = dense_flush_removezeros!(A.lnkmatrices, A.old_noderegions, A.sortednodesperthread, A.nt, A.rev_new_indices)
		end
	end
	A.phash = phash(A.cscmatrix)
	A.lnkmatrices = [SuperSparseMatrixLNK{matrixvaluetype(A), matrixindextype(A)}(A.n, A.nnts[tid]) for tid=1:A.nt]

end

"""
`CSC_RLNK_plusequals_less3_reordered_super!` from `plusequals.jl`
"""
function sparse_flush!(A::ExtendableSparseMatrixParallel; keep_zeros=true)

	#dropzeros!(
	plus_remap(A.lnkmatrices, A.cscmatrix, A.globalindices; keep_zeros)
	#)
		
end



"""
`CSC_RLNK_si_oc_ps_dz_less_reordered` from `conversion.jl`
"""
function dense_flush_keepzeros!(
	As::Vector{SuperSparseMatrixLNK{Tv, Ti}}, 
	onr, s, nt, rni
	) where {Tv, Ti <: Integer}

	nnz = sum([As[i].nnz for i=1:nt]) #you could also subtract the diagonal entries from shared columns, since those are definitely double
	indptr = zeros(Ti, As[1].m+1)
	indices = zeros(Ti, nnz) #sum(As.nnz))
	data = zeros(Float64, nnz) #sum(As.nnz))
	ctr = 1
	eqctr = 0
	tmp = zeros(Ti, size(onr)[1])
	
	#@warn [As[i].nnz for i=1:nt], [As[i].n for i=1:nt], [As[i].m for i=1:nt]
	#@info maximum.([As[i].colptr for i=1:nt])

	for nj=1:As[1].m
		indptr[nj] = ctr
		oj = rni[nj]
		regionctr = 1
		jc = 0
		nrr = view(onr, :, oj) 
		tmp .= 0
		for region in nrr #nrr #[:,j]
			regmod = region #(region-1)%nt+1
			if (region > 0) & !(region in tmp)
				k = s[regmod, nj]
				if regionctr == 1
					while k>0
						if As[regmod].rowval[k] != 0
							if ctr > nnz
								@info "ctr > nnz, $nj, $oj"
							end
							indices[ctr] = As[regmod].rowval[k]
							data[ctr]    = As[regmod].nzval[k]
							
							for jcc=1:jc
								if indices[ctr-jcc] > indices[ctr-jcc+1]
									tmp_i = indices[ctr-jcc+1]
									tmp_d = data[ctr-jcc+1]
									indices[ctr-jcc+1] = indices[ctr-jcc]
									data[ctr-jcc+1]    = data[ctr-jcc]
									
									indices[ctr-jcc] = tmp_i
									data[ctr-jcc]    = tmp_d
								else
									break
								end
							end
							
							ctr += 1
							jc += 1
						end
						k = As[regmod].colptr[k]
					end
				else
					while k>0
						if As[regmod].rowval[k] != 0
							indices[ctr] = As[regmod].rowval[k]
							data[ctr]    = As[regmod].nzval[k]
							
							for jcc=1:jc
								if indices[ctr-jcc] > indices[ctr-jcc+1]
									tmp_i = indices[ctr-jcc+1]
									tmp_d = data[ctr-jcc+1]
									indices[ctr-jcc+1] = indices[ctr-jcc]
									data[ctr-jcc+1]    = data[ctr-jcc]
									
									indices[ctr-jcc] = tmp_i
									data[ctr-jcc]    = tmp_d
								elseif indices[ctr-jcc] == indices[ctr-jcc+1]
									data[ctr-jcc] += data[ctr-jcc+1]
									eqctr += 1
									
									for jccc=1:jcc
										indices[ctr-jcc+jccc] = indices[ctr-jcc+jccc+1]
										data[ctr-jcc+jccc]    = data[ctr-jcc+jccc+1]
									end
									
									ctr -= 1
									jc  -= 1
									
									break
								else
									break
								end
							end
							
							ctr += 1
							jc += 1
						end
						k = As[regmod].colptr[k]
					end
					
				end
				tmp[regionctr] = region
				regionctr += 1
				
			end
			
		end
		
	end

	#@warn ctr/nnz
	
	indptr[end] = ctr
	resize!(indices, ctr-1)
	resize!(data, ctr-1)

	
	SparseArrays.SparseMatrixCSC(
		As[1].m, As[1].m, indptr, indices, data
	)
	
end


function dense_flush_removezeros!(
	As::Vector{SuperSparseMatrixLNK{Tv, Ti}}, 
	onr, s, nt, rni
	) where {Tv, Ti <: Integer}

	nnz = sum([As[i].nnz for i=1:nt]) #you could also subtract the diagonal entries from shared columns, since those are definitely double
	indptr = zeros(Ti, As[1].m+1)
	indices = zeros(Ti, nnz) #sum(As.nnz))
	data = zeros(Float64, nnz) #sum(As.nnz))
	ctr = 1
	eqctr = 0
	tmp = zeros(Ti, size(onr)[1])
	
	for nj=1:As[1].m
		indptr[nj] = ctr
		oj = rni[nj]
		regionctr = 1
		jc = 0
		nrr = view(onr, :, oj) 
		tmp .= 0
		for region in nrr #nrr #[:,j]
			regmod = region #(region-1)%nt+1
			if (region > 0) & !(region in tmp)
				k = s[regmod, nj]
				if regionctr == 1
					while k>0
						if As[regmod].nzval[k] != 0.0
							indices[ctr] = As[regmod].rowval[k]
							data[ctr]    = As[regmod].nzval[k]
							
							for jcc=1:jc
								if indices[ctr-jcc] > indices[ctr-jcc+1]
									tmp_i = indices[ctr-jcc+1]
									tmp_d = data[ctr-jcc+1]
									indices[ctr-jcc+1] = indices[ctr-jcc]
									data[ctr-jcc+1]    = data[ctr-jcc]
									
									indices[ctr-jcc] = tmp_i
									data[ctr-jcc]    = tmp_d
								else
									break
								end
							end
							
							ctr += 1
							jc += 1
						end
						k = As[regmod].colptr[k]
					end
				else
					while k>0
						if As[regmod].nzval[k] != 0.0
							indices[ctr] = As[regmod].rowval[k]
							data[ctr]    = As[regmod].nzval[k]
							
							for jcc=1:jc
								if indices[ctr-jcc] > indices[ctr-jcc+1]
									tmp_i = indices[ctr-jcc+1]
									tmp_d = data[ctr-jcc+1]
									indices[ctr-jcc+1] = indices[ctr-jcc]
									data[ctr-jcc+1]    = data[ctr-jcc]
									
									indices[ctr-jcc] = tmp_i
									data[ctr-jcc]    = tmp_d
								elseif indices[ctr-jcc] == indices[ctr-jcc+1]
									data[ctr-jcc] += data[ctr-jcc+1]
									eqctr += 1
									
									for jccc=1:jcc
										indices[ctr-jcc+jccc] = indices[ctr-jcc+jccc+1]
										data[ctr-jcc+jccc]    = data[ctr-jcc+jccc+1]
									end
									
									ctr -= 1
									jc  -= 1
									
									break
								else
									break
								end
							end
							
							ctr += 1
							jc += 1
						end
						k = As[regmod].colptr[k]
					end
					
				end
				tmp[regionctr] = region
				regionctr += 1
				
			end
			
		end
		
	end

	#@warn ctr/nnz
	
	indptr[end] = ctr
	resize!(indices, ctr-1)
	resize!(data, ctr-1)

	
	SparseArrays.SparseMatrixCSC(
		As[1].m, As[1].m, indptr, indices, data
	)
	
end







