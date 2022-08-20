"""
    LinearSolve.LinearProblem(A::ExtendableSparseMatrix,x,p=SciMLBase.NullParameters();u0=nothing,kwargs...)

Create linear problem from ExtendableSparseMatrix. This uses the internal SparseMatrixCSC, and thus allows to access
the functionality of [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl)
"""
function LinearSolve.LinearProblem(A::ExtendableSparseMatrix,x,p=SciMLBase.NullParameters();u0=nothing,kwargs...)
    flush!(A)
    LinearSolve.LinearProblem{false}(A.cscmatrix,x,p;u0,kwargs...)
end

"""
    LinearSolve.set_A(cache::LinearSolve.LinearCache, Aext::ExtendableSparseMatrix)

Update the linear solve cache from an ExtendableSparseMatrix. Note that this update allows
to take into account changes of the matrix pattern.
"""
function LinearSolve.set_A(cache::LinearSolve.LinearCache, Aext::ExtendableSparseMatrix)
    flush!(Aext)

    @unpack alg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose = cache

    if !pattern_equal(Aext.cscmatrix,A)
        cacheval = LinearSolve.init_cacheval(alg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)
        @set! cache.cacheval=cacheval
        @set! cache.isfresh = false
    end
    @set! cache.A = Aext.cscmatrix
    @set! cache.isfresh = true
    return cache
end
