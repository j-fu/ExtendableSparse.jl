### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 60941eaa-1aea-11eb-1277-97b991548781
begin 
	using Revise
    using Pkg
	Pkg.activate(joinpath(@__DIR__)
	using ExtendableSparse
	using BenchmarkTools
	using LinearSolve
	using LinearAlgebra
	using AlgebraicMultigrid
	using LinearSolvePardiso
	using PyPlot
end

# ╔═╡ 80247f26-bbc5-4253-a47b-26e953684850
md"""
## Calculations
"""

# ╔═╡ 6efa4657-93ed-4af8-a6af-aa3843e53857
small=false

# ╔═╡ b461d330-bf59-4984-995b-276b3df6d2ee
nsizes= small  ? 8 : 14

# ╔═╡ 9162f475-53a9-4de7-bd9a-57777e75707a
md"""
## 1D Unsymmetric
"""

# ╔═╡ 3d07fe0a-6153-4c5d-b649-a9d4f6ff9874
md"""
KLU wins over backlash (which is the same as umfpack)
"""

# ╔═╡ fad84a01-b79e-43a6-828d-4fe7bbc17f30
md"""
## 1D Symmetric
"""

# ╔═╡ 29ee574d-14ae-4d22-a4ab-75014cc1a7ab
md"""
Cholesky wins, and essentially is the same as backslash.
"""

# ╔═╡ c39a7edf-cbd6-4d1a-8a07-51b0e5700ea8
md"""
## 2D Unsymmetric
"""

# ╔═╡ 5ed0a15b-d1d0-4e84-81a1-154fac35f98f
md"""
Backslash wins which is the same as umfpack
"""

# ╔═╡ f3ff8efb-6407-4701-bcf7-7798c8494447
md"""
## 2D Symmetric
"""

# ╔═╡ c075afaf-fbd9-4c34-b5df-363e16d93497
md"""
Cholesky/backslash wins
"""

# ╔═╡ eb40d970-ca0f-4216-bd19-80fdaf01b918
md"""
## 3D unsymmetric
"""

# ╔═╡ 59d94db2-d554-4a82-afbf-d10ed740ab2c
md"""
gmres wins..., next comes Pardiso
"""

# ╔═╡ 62cd5493-ab52-4826-9467-a34d88782ecd
md"""
## 3D Symmetric
"""

# ╔═╡ d19121c2-18bb-422e-a04e-18c50cc2b1af
md"""
pcg wins, and of the factorizations, Pardiso wins for large problems, otherwise Cholesky/backslash.
"""

# ╔═╡ d7200524-6365-42f6-a096-77484df9e247
md"""
## Library
"""

# ╔═╡ f991f70e-b202-4e0a-86bf-2c7cd7dbb678
function solverbenchmarks(solver;
          dims=[1,2,3],
          nsizes=8,
          symmetric=[true,false],
          matrixtype=ExtendableSparseMatrix,
          prefix="",
          tol=sqrt(eps(Float64)))
	result=Dict{Symbol,Any}()	
	for dim in dims
		for symm in symmetric
			name= Symbol(prefix*(symm ? "_s_" : "_u_")*string(dim)*"d")
			result[name]=solverbenchmark(solver;dim,nsizes,symmetric=symm,matrixtype,tol)
		end
	end
	sort(result)
end

# ╔═╡ c48ab047-237a-4f58-a363-348437bff8f4
r_backslash=solverbenchmarks(;nsizes,prefix="backslash",tol=1.0e-5) do  A,b
	A\b
end

# ╔═╡ ffcae8d1-f110-4465-a865-d7f9eb7a0866
r_umfpack=solverbenchmarks(;nsizes, symmetric=[false],prefix="umfpack") do  A,b
	 solve(LinearProblem(A,b),UMFPACKFactorization())
end

# ╔═╡ fae66489-0e7f-4dd3-b26e-9097e101f614
r_cholesky=solverbenchmarks(;nsizes,symmetric=[true],prefix="cholesky",tol=1.0e-5) do  A,b
	 factorization=CholeskyFactorization()
factorize!(factorization,A)
factorization\b
end

# ╔═╡ 6f0cc88d-af70-4638-ae2d-813f52bda635
r_klu=solverbenchmarks(;nsizes=min(nsizes,11),prefix="klu", symmetric=[false]) do  A,b
	 solve(LinearProblem(A,b),KLUFactorization())
end

# ╔═╡ 5e74214a-3964-4777-a4d1-5da8c0bd58d3
r_pcg=solverbenchmarks(;nsizes,dims=[2,3],symmetric=[true],prefix="pcg",tol=1.0e-2) do  A,b
	 solve(LinearProblem(A,b),KrylovJL_CG(),Pl=ILU0Preconditioner(A),reltol=1.0e-10)
end

# ╔═╡ c92d6e0a-e938-4090-9b41-03c9aeed4d0e
r_pgmres=solverbenchmarks(;nsizes,dims=(3),symmetric=[false],prefix="pgmres",tol=1.0e-5) do  A,b
	 solve(LinearProblem(A,b),KrylovKitJL_GMRES(),Pl=JacobiPreconditioner(A),
		 abstol=1.0e-10,maxiters=10000)
end

# ╔═╡ 6d58399c-1590-4ab5-a712-01b67ffabbfe
r_mklpardiso=solverbenchmarks(;nsizes,dims=(1,2,3),symmetric=[false],prefix="mklpardiso",tol=1.0e-5) do  A,b
factorization=MKLPardisoLU()
factorize!(factorization,A)
factorization\b
end

# ╔═╡ 76ea5712-bdf3-4180-b18e-87702c661f89
r_lspardiso=solverbenchmarks(;nsizes,dims=(1,2,3),symmetric=[false],prefix="lspardiso") do  A,b
solve(LinearProblem(A,b),MKLPardisoFactorize(nprocs=5))
end

# ╔═╡ 7932e78e-892b-40b0-8067-5d3226d5e005
alldicts=merge(r_umfpack,r_klu, r_lspardiso, r_backslash,r_pgmres,r_pcg,r_cholesky)

# ╔═╡ 6a72395d-4f41-4f12-abd7-501575feaf84
function pyplot(results;title="", subset=nothing)
    clf()
	if subset!=nothing
		results=filter(p->first(p) ∈ subset,results)
	end		
    N=[1]
	for result in results
		loglog(last(result)...,label=string(first(result)))
	end
	x0=minimum(r->last(r)[1][1],results)
	xend=maximum(r->last(r)[1][end],results)
	y0=minimum(r->last(r)[2][1],results)
	N=[x0,xend]
	fac=0.5*y0/x0
	PyPlot.title(title)
	loglog(N,fac*N,"k--",label="O(N)")
#	loglog(N1d,1.0e-7*N1d.^2,"k-.",label="O(N^2)")
	legend(loc=("upper left"))
	grid()
	gcf().set_size_inches(6,4)
	gcf()
end

# ╔═╡ 1c27cbeb-13a6-4f48-ab2c-d1cc815a5f7e
pyplot(r_backslash,title="backslash")

# ╔═╡ e49fabfd-fc3d-4acf-9939-3eb5d7253def
pyplot(sort(merge(r_umfpack,r_cholesky)),title="umfpack+cholesky")

# ╔═╡ 48c855c8-29d8-42ba-a674-65ca315780ee
pyplot(sort(merge(r_pcg,r_cholesky)),title="pcg+cholesky")

# ╔═╡ d5fc4357-d4de-4800-990e-da8d57043ecd
pyplot(r_klu,title="klu")

# ╔═╡ 6fd7617b-d420-4574-8981-f93f6d2d0149
pyplot(merge(r_pgmres,r_umfpack)|>sort,title="gmres+umfpack")

# ╔═╡ e3788d1b-1328-4f7c-8b86-04ddb835ae10
pyplot(merge(r_lspardiso,r_umfpack)|>sort)

# ╔═╡ 5f60047b-8ac5-476a-8253-0de9b36215ab
pyplot(alldicts,subset=(:klu_u_1d, :umfpack_u_1d, :lspardiso_u_1d, :backslash_u_1d))

# ╔═╡ ac591b38-76c4-4784-9f0a-9ad4d7c2d594
pyplot(alldicts,subset=(:klu_u_1d, :umfpack_u_1d, :lspardiso_u_1d, :backslash_s_1d,:cholesky_s_1d))

# ╔═╡ 329273d0-f375-4c1d-a2db-f1ef87f5aaaf
pyplot(alldicts,subset=(:klu_u_2d, :umfpack_u_2d, :lspardiso_u_2d,:backslash_u_2d))

# ╔═╡ 2a9d1f88-9a5c-4063-8cc8-c91822ae83f4
pyplot(alldicts,subset=(:klu_u_2d, :umfpack_u_2d, :lspardiso_u_2d,:backslash_s_2d, :cholesky_s_2d))

# ╔═╡ d6c2dcb8-3221-41c4-be50-3b83c8776f21
pyplot(alldicts, subset=(:klu_u_3d, :umfpack_u_3d, :lspardiso_u_3d, :backslash_u_3d,:pgmres_u_3d))

# ╔═╡ 318b60ec-d778-4e59-a699-cdc4fd42bf36
pyplot(alldicts, subset=(:klu_u_3d, :umfpack_u_3d, :lspardiso_u_3d, :backslash_s_3d,:pcg_s_3d,:cholesky_s_3d))

# ╔═╡ Cell order:
# ╠═60941eaa-1aea-11eb-1277-97b991548781
# ╠═80247f26-bbc5-4253-a47b-26e953684850
# ╠═6efa4657-93ed-4af8-a6af-aa3843e53857
# ╠═b461d330-bf59-4984-995b-276b3df6d2ee
# ╠═c48ab047-237a-4f58-a363-348437bff8f4
# ╠═ffcae8d1-f110-4465-a865-d7f9eb7a0866
# ╠═fae66489-0e7f-4dd3-b26e-9097e101f614
# ╠═6f0cc88d-af70-4638-ae2d-813f52bda635
# ╠═5e74214a-3964-4777-a4d1-5da8c0bd58d3
# ╠═1c27cbeb-13a6-4f48-ab2c-d1cc815a5f7e
# ╠═e49fabfd-fc3d-4acf-9939-3eb5d7253def
# ╠═48c855c8-29d8-42ba-a674-65ca315780ee
# ╠═d5fc4357-d4de-4800-990e-da8d57043ecd
# ╠═c92d6e0a-e938-4090-9b41-03c9aeed4d0e
# ╠═6fd7617b-d420-4574-8981-f93f6d2d0149
# ╠═6d58399c-1590-4ab5-a712-01b67ffabbfe
# ╠═76ea5712-bdf3-4180-b18e-87702c661f89
# ╠═e3788d1b-1328-4f7c-8b86-04ddb835ae10
# ╠═7932e78e-892b-40b0-8067-5d3226d5e005
# ╠═9162f475-53a9-4de7-bd9a-57777e75707a
# ╠═5f60047b-8ac5-476a-8253-0de9b36215ab
# ╠═3d07fe0a-6153-4c5d-b649-a9d4f6ff9874
# ╠═fad84a01-b79e-43a6-828d-4fe7bbc17f30
# ╠═ac591b38-76c4-4784-9f0a-9ad4d7c2d594
# ╠═29ee574d-14ae-4d22-a4ab-75014cc1a7ab
# ╠═c39a7edf-cbd6-4d1a-8a07-51b0e5700ea8
# ╠═329273d0-f375-4c1d-a2db-f1ef87f5aaaf
# ╠═5ed0a15b-d1d0-4e84-81a1-154fac35f98f
# ╠═f3ff8efb-6407-4701-bcf7-7798c8494447
# ╠═2a9d1f88-9a5c-4063-8cc8-c91822ae83f4
# ╠═c075afaf-fbd9-4c34-b5df-363e16d93497
# ╠═eb40d970-ca0f-4216-bd19-80fdaf01b918
# ╠═d6c2dcb8-3221-41c4-be50-3b83c8776f21
# ╠═59d94db2-d554-4a82-afbf-d10ed740ab2c
# ╠═62cd5493-ab52-4826-9467-a34d88782ecd
# ╠═318b60ec-d778-4e59-a699-cdc4fd42bf36
# ╠═d19121c2-18bb-422e-a04e-18c50cc2b1af
# ╠═d7200524-6365-42f6-a096-77484df9e247
# ╠═f991f70e-b202-4e0a-86bf-2c7cd7dbb678
# ╠═6a72395d-4f41-4f12-abd7-501575feaf84
