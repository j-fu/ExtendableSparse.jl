module ExperimentalParallel

using ExtendableSparse,SparseArrays
using ExtendableSparse.Experimental
using BenchmarkTools
using OhMyThreads: @tasks
using Test

import ChunkSplitters
# Methods to test parallel assembly
# Will eventually become part of the package.

"""

Return colored partitioing of grid made up by `X` and `Y`  for work with `max(nt,4)` threads
as a vector `p` of a vector pairs of index ranges such that `p[i]` containes partions
of color i which can be assembled independently.

The current algorithm  creates `nt^2` partitions with `nt` colors.
"""
function part2d(X,Y, nt)
    nt=max(4,nt)
    XP=collect(ChunkSplitters.chunks(1:length(X)-1,n=nt))
    YP=collect(ChunkSplitters.chunks(1:length(Y)-1,n=nt))
    partitions = [Tuple{StepRange{Int64}, StepRange{Int64}}[] for i = 1:nt]
    ipart=1
    col=1
    for jp=1:nt
        for ip=1:nt
            push!(partitions[col], (XP[ip], YP[jp]))
            col=(col -1 +1 )%nt+1
        end
        col=(col -1 +2)%nt+1
    end
    partitions
end

function colpart2d(X,Y,nt)
    Nx=length(X)
    Ny=length(Y)
    p=part2d(X,Y,nt)
    pc=zeros(Int,sum(length,p))
    jp=1
    for icol=1:length(p)
        for ip=1:length(p[icol])
            pc[jp]=icol
            jp+=1
        end
    end
    p,pc
end


"""
    showgrid(Makie, ColorSchemes, X,Y,nt)

Show grid partitioned according to [`part2d`](@ref). Needs a makie variant and ColorSchemes
to be passed as modules.
"""
function showgrid(Makie, ColorSchemes, X,Y,nt)
    f = Makie.Figure()
    ax = Makie.Axis(f[1, 1]; aspect = 1)
    p=part2d(X,Y,nt)
    ncol=length(p)
    @show sum(length,p), ncol
    colors=get(ColorSchemes.rainbow,collect(1:ncol)/ncol)
    poly=Vector{Makie.Point2f}(undef,4)
    for icol = 1:ncol
        for (xp, yp) in p[icol]
            for j in yp
                for i in xp
                    poly[1]=Makie.Point2f(X[i], Y[j])
                    poly[2]=Makie.Point2f(X[i + 1], Y[j])
                    poly[3]=Makie.Point2f(X[i + 1], Y[j + 1])
                    poly[4]=Makie.Point2f(X[i], Y[j + 1])
                    Makie.poly!(copy(poly),color = colors[icol])
                end
            end
        end
    end
    f
end


"""

Assemble edge for finite volume laplacian.
Used by [`partassemble!`](@ref).
"""
function assembleedge!(A,v,k,l)
    rawupdateindex!(A,+,v,k,k)
    rawupdateindex!(A,+,-v,k,l)
    rawupdateindex!(A,+,-v,l,k)
    rawupdateindex!(A,+,v,l,l)
end

function assembleedge!(A,v,k,l,tid)
    rawupdateindex!(A,+,v,k,k,tid)
    rawupdateindex!(A,+,-v,k,l,tid)
    rawupdateindex!(A,+,-v,l,k,tid)
    rawupdateindex!(A,+,v,l,l,tid)
end

"""
Assemble finite volume Laplacian + diagnonal term
on grid cell `i,j`.
Used by [`partassemble!`](@ref).
"""
function assemblecell!(A,lindexes,X,Y,i,j,d)
    hx=X[i+1]-X[i]
    hy=Y[j+1]-Y[j]
    ij00=lindexes[i,j]
    ij10=lindexes[i+1,j]
    ij11=lindexes[i+1,j+1]
    ij01=lindexes[i,j+1]
    
    assembleedge!(A,0.5*hx/hy,ij00,ij01)
    assembleedge!(A,0.5*hx/hy,ij10,ij11)
    assembleedge!(A,0.5*hy/hx,ij00,ij10)
    assembleedge!(A,0.5*hy/hx,ij01,ij11)
    v=0.25*hx*hy
    rawupdateindex!(A,+,v*d,ij00,ij00)
    rawupdateindex!(A,+,v*d,ij01,ij01)
    rawupdateindex!(A,+,v*d,ij10,ij10)
    rawupdateindex!(A,+,v*d,ij11,ij11)
end

function assemblecell!(A,lindexes,X,Y,i,j,d,tid)
    hx=X[i+1]-X[i]
    hy=Y[j+1]-Y[j]
    ij00=lindexes[i,j]
    ij10=lindexes[i+1,j]
    ij11=lindexes[i+1,j+1]
    ij01=lindexes[i,j+1]
    
    assembleedge!(A,0.5*hx/hy,ij00,ij01,tid)
    assembleedge!(A,0.5*hx/hy,ij10,ij11,tid)
    assembleedge!(A,0.5*hy/hx,ij00,ij10,tid)
    assembleedge!(A,0.5*hy/hx,ij01,ij11,tid)
    v=0.25*hx*hy
    rawupdateindex!(A,+,v*d,ij00,ij00,tid)
    rawupdateindex!(A,+,v*d,ij01,ij01,tid)
    rawupdateindex!(A,+,v*d,ij10,ij10,tid)
    rawupdateindex!(A,+,v*d,ij11,ij11,tid)
end

"""

Assemble finite volume Laplacian + diagnonal term
on grid cells in partition described by ranges xp,yp.
Used by [`partassemble!`](@ref).
"""
function assemblepartition!(A,lindexes,X,Y,xp,yp,d)
    for j in yp
	for i in xp
	    assemblecell!(A,lindexes,X,Y,i,j,d)
	end
    end
end

function assemblepartition!(A,lindexes,X,Y,xp,yp,d,tid)
    for j in yp
	for i in xp
	    assemblecell!(A,lindexes,X,Y,i,j,d,tid)
	end
    end
end

"""
    partassemble!(A,N,nt=1;xrange=(0,1),yrange=(0,1), d=0.1)

Partitioned, cellwise, multithreaded assembly of finite difference matrix for
` -Δu + d*u=f` with homogeneous Neumann bc on grid  set up by coordinate vectors
`X` and `Y` partitioned for work with `nt` threads
Does not work during structure setup.
"""
function partassemble!(A,X,Y,nt=1;d=0.1)
    Nx=length(X)
    Ny=length(Y)
    size(A,1)==Nx*Ny || error("incompatible size of A")
    size(A,2)==Nx*Ny || error("incompatible size of A")

    lindexes=LinearIndices((1:Nx,1:Ny))
    if nt==1
        assemblepartition!(A,lindexes,X,Y,1:Nx-1,1:Nx-1,d)
    else
        p=part2d(X,Y,nt)
        for icol=1:length(p)
	    @tasks for (xp, yp) in p[icol]
	        assemblepartition!(A,lindexes,X,Y,xp,yp,d)
	    end
        end
    end
    flush!(A)
end


function partassemble!(A::Union{MTExtendableSparseMatrixCSC},X,Y,nt=1;d=0.1, reset=true)
    Nx=length(X)
    Ny=length(Y)

    size(A,1)==Nx*Ny || error("incompatible size of A")
    size(A,2)==Nx*Ny || error("incompatible size of A")

    lindexes=LinearIndices((1:Nx,1:Ny))
    if nt==1
        reset!(A,1)
        assemblepartition!(A,lindexes,X,Y,1:Nx-1,1:Nx-1,d,1)
    else
        p,pc=colpart2d(X,Y,nt)
        if reset
            reset!(A,pc)
        end
        jp0=0
        for icol=1:length(p)
            npc=length(p[icol])
	    @tasks for ip=1:npc
                (xp, yp)=p[icol][ip]
	        assemblepartition!(A,lindexes,X,Y,xp,yp,d,jp0+ip)
	    end
            jp0+=npc
        end
    end
    flush!(A)
end




"""
`test_ESMP(n, nt; depth=1, Tv=Float64, Ti=Int64, k=10)`

Measure and output times for build and update for a rectangle grid with `n * n` cells.
Calculations are done on `nt` threads (`nt` >= 1).
Returns the assembled matrix.
"""
function test_ESMP(n, nt; depth=1, Tv=Float64, Ti=Int64, k=10)
    m = n
    lindexes = LinearIndices((1:n,1:m))
    mat_cell_node, nc, nn = generate_rectangle_grid(lindexes, Ti)
    if nt > 1
        A = ExtendableSparseMatrixParallel{Tv, Ti}(mat_cell_node, nc, nn, nt, depth; block_struct=false)
    else
        A = ExtendableSparseMatrix{Tv, Ti}(n*m, n*m)
    end

    X = collect(1:n) #LinRange(0,1,n)
    Y = collect(1:n) #LinRange(0,1,m)

    #Build
    times_build = zeros(k)
    for i=1:k
        ExtendableSparse.reset!(A)
        times_build[i] = @elapsed assemble_ESMP(A, n-1, m-1, mat_cell_node, X, Y; set_CSC_zero=false)
    end



    #update
    times_update = zeros(k)
    for i=1:k
        times_update[i] = @elapsed assemble_ESMP(A, n-1, m-1, mat_cell_node, X, Y; set_CSC_zero=true)
    end

    @info "TIMES:  MIN,  AVG,  MAX"
    info_minmax(times_build, "build ")
    info_minmax(times_update, "update")
    
    A
end


function test_correctness_build(n, depth=1, Tv=Float64, Ti=Int64, allnp=[4,5,6,7,8,9,10])
    m = n
    lindexes = LinearIndices((1:n,1:m))
    X = collect(1:n) #LinRange(0,1,n)
    Y = collect(1:n) #LinRange(0,1,m)
    
    mat_cell_node, nc, nn = generate_rectangle_grid(lindexes, Ti)
    
    A0 = ExtendableSparseMatrix{Tv, Ti}(n*m, n*m)
    assemble_ESMP(A0, n-1, m-1, mat_cell_node, X, Y; set_CSC_zero=false)

    for nt in allnp
        A = ExtendableSparseMatrixParallel{Tv, Ti}(mat_cell_node, nc, nn, nt, depth; block_struct=false)
        assemble_ESMP(A, n-1, m-1, mat_cell_node, X, Y; set_CSC_zero=false)
        @assert A.cscmatrix≈A0.cscmatrix
    end
end


function speedup_build(n, depth=1, Tv=Float64, Ti=Int64, allnp=[4,5,6,7,8,9,10])
    m = n
    lindexes = LinearIndices((1:n,1:m))
    X = collect(1:n) #LinRange(0,1,n)
    Y = collect(1:n) #LinRange(0,1,m)

    mat_cell_node, nc, nn = generate_rectangle_grid(lindexes, Ti)

    A0 = ExtendableSparseMatrix{Tv, Ti}(n*m, n*m)
    t0=@belapsed assemble_ESMP($A0, $n-1, $m-1, $mat_cell_node, $X, $Y; set_CSC_zero=false) seconds=1 setup=(reset!($A0))
    
    result=[]

    for nt in allnp
        A = ExtendableSparseMatrixParallel{Tv, Ti}(mat_cell_node, nc, nn, nt, depth; block_struct=false)
        t=@belapsed assemble_ESMP($A, $n-1, $m-1, $mat_cell_node, $X, $Y; set_CSC_zero=false) setup=(ExtendableSparse.reset!($A)) seconds=1
        @assert A.cscmatrix≈A0.cscmatrix
        push!(result,(nt,round(t0/t,digits=2)))
    end

    result
    
end


function speedup_update(n, depth=1, Tv=Float64, Ti=Int64, allnp=[4,5,6,7,8,9,10])
    m = n
    lindexes = LinearIndices((1:n,1:m))
    X = collect(1:n) #LinRange(0,1,n)
    Y = collect(1:n) #LinRange(0,1,m)

    mat_cell_node, nc, nn = generate_rectangle_grid(lindexes, Ti)

    A0 = ExtendableSparseMatrix{Tv, Ti}(n*m, n*m)
    assemble_ESMP(A0, n-1, m-1, mat_cell_node, X, Y)
    t0=@belapsed assemble_ESMP($A0, $n-1, $m-1, $mat_cell_node, $X, $Y; set_CSC_zero=false) seconds=1 setup=(nonzeros($A0.cscmatrix).=0)
    


    result=[]

    for nt in allnp
        A = ExtendableSparseMatrixParallel{Tv, Ti}(mat_cell_node, nc, nn, nt, depth; block_struct=false)
        assemble_ESMP(A, n-1, m-1, mat_cell_node, X, Y; set_CSC_zero=false)
        t=@belapsed assemble_ESMP($A, $n-1, $m-1, $mat_cell_node, $X, $Y; set_CSC_zero=false) setup=(nonzeros($A.cscmatrix).=0) seconds=1
        @assert A.cscmatrix≈A0.cscmatrix
        push!(result,(nt,round(t0/t,digits=2)))
    end

    result
    
end


"""
`generate_rectangle_grid(lindexes, Ti)`

Generate a rectangle grid (i.e. a CellNodes matrix) based on LinerIndices
"""
function generate_rectangle_grid(lindexes, Ti)
    n,m = size(lindexes)
    nn = n*m # num nodes
    nc = (n-1)*(m-1)
    #lindexes=LinearIndices((1:n,1:m))

    mat_cell_node = zeros(Ti, 4, nc)

    # links oben, rechts oben, rechts unten, links unten
    cell_id = 1
    for ir in 1:n-1
        for jr in 1:m-1
            mat_cell_node[1,cell_id] = lindexes[ir,jr]
            mat_cell_node[2,cell_id] = lindexes[ir,jr+1]
            mat_cell_node[3,cell_id] = lindexes[ir+1,jr+1]
            mat_cell_node[4,cell_id] = lindexes[ir+1,jr]
            cell_id += 1
        end
    end


    mat_cell_node, nc, nn

end

function info_minmax(x, name; digits=3)
    n = length(x)
    @info name*" $(round(minimum(x),digits=digits)), $(round(sum(x)/n,digits=digits)), $(round(maximum(x),digits=digits))"
end

"""
Assembly functions for ExtendableSparseMatrixParallel
"""
function assemble_ESMP(A::ExtendableSparseMatrixParallel{Tv, Ti}, n, m, mat_cell_node, X, Y; d=0.1, set_CSC_zero=true) where {Tv, Ti <: Integer}
    if set_CSC_zero
        A.cscmatrix.nzval .= 0
    end

    for level=1:A.depth
        @tasks for tid=1:A.nt
            for cell in A.cellsforpart[(level-1)*A.nt+tid]
                assemblecell!(A, n, m, mat_cell_node, X, Y, d, cell, tid)
            end
        end
    end

    for cell in A.cellsforpart[A.depth*A.nt+1]
        assemblecell!(A, n, m, mat_cell_node, X, Y, d, cell, 1)
    end
    nnzCSC, nnzLNK = nnz_noflush(A)
    if nnzCSC > 0 && nnzLNK > 0	
        flush!(A; do_dense=false)
        #sparse flush
    elseif nnzCSC == 0 && nnzLNK > 0
        flush!(A; do_dense=true)
        #dense flush
    end
end


function assembleedge!(A::ExtendableSparseMatrixParallel{Tv, Ti},v,k,l,tid) where {Tv, Ti <: Integer}
    addtoentry!(A, k, k, tid, +v)
    addtoentry!(A, k, l, tid, -v)
    addtoentry!(A, l, k, tid, -v)
    addtoentry!(A, l, l, tid, +v)
end

function assemblecell!(A::ExtendableSparseMatrixParallel{Tv, Ti},n,m,mcn,X,Y,d,cell,tid) where {Tv, Ti <: Integer}
    ij00=mcn[1,cell]
    ij10=mcn[2,cell]
    ij11=mcn[3,cell]
    ij01=mcn[4,cell]
    
    ix = (cell-1)%n+1
    iy = Int64(ceil(cell/n))

    hx=X[ix+1]-X[ix]
    hy=Y[iy+1]-Y[iy]
    
    assembleedge!(A,0.5*hx/hy,ij00,ij01,tid)
    assembleedge!(A,0.5*hx/hy,ij10,ij11,tid)
    assembleedge!(A,0.5*hy/hx,ij00,ij10,tid)
    assembleedge!(A,0.5*hy/hx,ij01,ij11,tid)
    v=0.25*hx*hy
    addtoentry!(A, ij00, ij00, tid, v*d)
    addtoentry!(A, ij01, ij01, tid, v*d)
    addtoentry!(A, ij10, ij10, tid, v*d)
    addtoentry!(A, ij11, ij11, tid, v*d)
end



"""
Assembly functions for ExtendableSparseMatrix
"""
function assemble_ESMP(A::ExtendableSparseMatrix{Tv, Ti}, n, m, mat_cell_node, X, Y; d=0.1, set_CSC_zero=true) where {Tv, Ti <: Integer}
    if set_CSC_zero
        A.cscmatrix.nzval .= 0
    end
    nc = size(mat_cell_node,2)
    for cell=1:nc
        assemblecell!(A, n, m, mat_cell_node, X, Y, d, cell)
    end
    ExtendableSparse.flush!(A) 
end

function assembleedge!(A::ExtendableSparseMatrix{Tv, Ti},v,k,l) where {Tv, Ti <: Integer}
    A[k,k]+=v
    A[k,l]-=v
    A[l,k]-=v
    A[l,l]+=v
end

function assemblecell!(A::ExtendableSparseMatrix{Tv, Ti},n,m,mcn,X,Y,d,cell) where {Tv, Ti <: Integer}
    ij00=mcn[1,cell]
    ij10=mcn[2,cell]
    ij11=mcn[3,cell]
    ij01=mcn[4,cell]
    
    ix = (cell-1)%n+1
    iy = Int64(ceil(cell/n))

    hx=X[ix+1]-X[ix]
    hy=Y[iy+1]-Y[iy]
    
    assembleedge!(A,0.5*hx/hy,ij00,ij01)
    assembleedge!(A,0.5*hx/hy,ij10,ij11)
    assembleedge!(A,0.5*hy/hx,ij00,ij10)
    assembleedge!(A,0.5*hy/hx,ij01,ij11)
    v=0.25*hx*hy
    A[ij00,ij00]+=v*d
    A[ij01,ij01]+=v*d
    A[ij10,ij10]+=v*d
    A[ij11,ij11]+=v*d
end
end
