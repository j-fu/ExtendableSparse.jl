module ExperimentalParallel

using ExtendableSparse,SparseArrays
using ExtendableSparse.Experimental
using BenchmarkTools
using OhMyThreads: @tasks
using Test


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
