import ChunkSplitters
# Methods to test parallel assembly
# Will eventually become part of the package.

"""
    $(SIGNATURES)

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
    $(SIGNATURES)

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
    $(SIGNATURES)

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
    $(SIGNATURES)

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


function partassemble!(A::Union{ExtendableSparseMatrixParallelDict,ExtendableSparseMatrixParallelLNKDict,ExtendableSparseMatrixParallelLNKX},X,Y,nt=1;d=0.1, reset=true)
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
