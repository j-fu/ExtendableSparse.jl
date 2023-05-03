# Integration with LinearSolve.jl

Starting with version 0.9.6, ExtendableSparse is compatible
with [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl).
Since version 0.9.7, this is facilitated via the
AbstractSparseMatrixCSC interface.

```@autodocs
Modules = [ExtendableSparse]
Pages = ["linearsolve.jl"]
```

We can create a test problem and solve it with the `\` operator.

```@example
using ExtendableSparse # hide
A = fdrand(10, 10, 10; matrixtype = ExtendableSparseMatrix)
x = ones(1000)
b = A * x
y = A \ b
sum(y)
```

The same problem can be solved by the tools available via `LinearSolve.jl`:

```@example
using ExtendableSparse # hide
using LinearSolve # hide
A = fdrand(10, 10, 10; matrixtype = ExtendableSparseMatrix)
x = ones(1000)
b = A * x
y = solve(LinearProblem(A, b), SparspakFactorization()).u
sum(y)
```

Also, the iterative method interface works with ExtendableSparse.

```@example
using ExtendableSparse # hide
using LinearSolve # hide
using SparseArrays # hide
using ILUZero # hide
A = fdrand(10, 10, 10; matrixtype = ExtendableSparseMatrix)
x = ones(1000)
b = A * x
y = LinearSolve.solve(LinearProblem(A, b), KrylovJL_CG();
                      Pl = ILUZero.ilu0(SparseMatrixCSC(A))).u
sum(y)
```

However, ExtendableSparse provides a number of wrappers around preconditioners
from various Julia packages.
```@example
using ExtendableSparse # hide
using LinearSolve # hide
using ILUZero # hide
A = fdrand(10, 10, 10; matrixtype = ExtendableSparseMatrix)
x = ones(1000)
b = A * x
y = LinearSolve.solve(LinearProblem(A, b), KrylovJL_CG();
                      Pl = ILUZeroPreconditioner(A)).u
sum(y)
```
