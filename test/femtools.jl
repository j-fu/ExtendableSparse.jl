using OhMyThreads: @tasks
using RecursiveFactorization

function testgrid(N; dim=3)
    X = range(0, 1; length=N^(1.0 / dim) |> ceil |> Int)
    simplexgrid((X for i = 1:dim)...)
end

function coordmatrix!(C, coord, cellnodes, k)
    spacedim=size(coord,1)
    celldim=size(cellnodes,1)
    @inbounds for jj = 1:celldim
        C[1, jj] = 1
        @inbounds for ii = 1:spacedim
            C[ii + 1, jj] = coord[ii, cellnodes[jj, k]]
        end
    end
end

function gradient!(G, C, factdim, I, ipiv)
    clu = RecursiveFactorization.lu!(C, ipiv, Val(true), Val(false))
    ldiv!(G, clu, I)
    abs(det(clu)) / factdim
end

function scalpro(G, dim, jl, il)
    s = 0.0
    @inbounds @simd for k = 1:dim
        s += G[jl, k + 1] * G[il, k + 1]
    end
    return s
end

function stiffness!(S, dim, G)
    @inbounds for il = 1:(dim + 1)
        S[il, il] = scalpro(G, dim, il, il)
        @inbounds for jl = (il + 1):(dim + 1)
            S[il, jl] = scalpro(G, dim, jl, il)
            S[jl, il] = S[il, jl]
        end
    end
    return S
end

function testassemble!(A_h, grid)
    coord = grid[Coordinates]
    cellnodes = grid[CellNodes]
    ncells = num_cells(grid)
    dim = size(coord, 1)
    lnodes = dim + 1
    factdim::Float64 = factorial(dim)
    S = zeros(lnodes, lnodes) # local stiffness matrix
    C = zeros(lnodes, lnodes)  # local coordinate matrix
    G = zeros(lnodes, lnodes) # shape function gradients
    ipiv = zeros(Int, lnodes)
    I = Matrix(Diagonal(ones(lnodes)))
    ncells = size(cellnodes, 2)
    for icell = 1:ncells
        coordmatrix!(C, coord, cellnodes, icell)
        vol = gradient!(G, C, factdim, I, ipiv)
        stiffness!(S, dim, G)
        for il = 1:lnodes
            i = cellnodes[il, icell]
            rawupdateindex!(A_h, +, 0.1 * vol / (dim + 1), i, i)
            for jl = 1:lnodes
                j = cellnodes[jl, icell]
                rawupdateindex!(A_h, +, vol * (S[il, jl]), i, j)
            end
        end
    end
    flush!(A_h)
end

function testassemble_parallel!(A_h, grid)
    coord = grid[Coordinates]
    cellnodes = grid[CellNodes]
    ncells = num_cells(grid)
    dim = size(coord, 1)
    lnodes = dim + 1
    npart = num_partitions(grid)
    factdim::Float64 = factorial(dim)
    SS = [zeros(lnodes, lnodes) for i = 1:npart] # local stiffness matrix
    CC = [zeros(lnodes, lnodes) for i = 1:npart] # local coordinate matrix
    GG = [zeros(lnodes, lnodes) for i = 1:npart] # shape function gradients
    IP = [zeros(Int, lnodes) for i = 1:npart] # shape function gradients
    I = Matrix(Diagonal(ones(lnodes)))
    ncells = size(cellnodes, 2)
    for color in pcolors(grid)
        @tasks for part in pcolor_partitions(grid, color)
            C = CC[part]
            S = SS[part]
            G = GG[part]
            ipiv = IP[part]
            for icell in partition_cells(grid, part)
                coordmatrix!(C, coord, cellnodes, icell)
                vol = gradient!(G, C, factdim, I, ipiv)
                stiffness!(S, dim, G)
                for il = 1:lnodes
                    i = cellnodes[il, icell]
                    rawupdateindex!(A_h, +, 0.1 * vol / (dim + 1), i, i, part)
                    for jl = 1:lnodes
                        j = cellnodes[jl, icell]
                        rawupdateindex!(A_h, +, vol * (S[il, jl]), i, j, part)
                    end
                end
            end
        end
    end
    flush!(A_h)
end
