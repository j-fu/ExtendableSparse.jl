"""
$(SIGNATURES)

Update LinearSolve cache.
"""
LinearSolve.set_A(cache::LinearSolve.LinearCache, A::ExtendableSparseMatrix) = LinearSolve.set_A(cache, sparse(A))

"""
$(SIGNATURES)

Create LinearProblem from ExtendableSparseMatrix.
"""
LinearSolve.LinearProblem(A::ExtendableSparseMatrix,b)=LinearSolve.LinearProblem(sparse(A),b)

