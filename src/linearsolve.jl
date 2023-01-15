"""
$(SIGNATURES)

Update LinearSolve cache.
"""
function LinearSolve.set_A(cache::LinearSolve.LinearCache, A::ExtendableSparseMatrix)
    LinearSolve.set_A(cache, sparse(A))
end

"""
$(SIGNATURES)

Create LinearProblem from ExtendableSparseMatrix.
"""
function LinearSolve.LinearProblem(A::ExtendableSparseMatrix, b)
    LinearSolve.LinearProblem(sparse(A), b)
end
