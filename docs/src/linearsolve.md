# Integration with LinearSolve.jl

Starting with version 0.7, ExtendableSparse tries to become compatible
with [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl). 
For this purpose, it extends the `LinearProblem` constructor and the
`set_A` function by methods specific for `ExtendableSparseMatrix`.

```@docs
LinearSolve.LinearProblem(::ExtendableSparseMatrix,x,p;kwargs...)
LinearSolve.set_A(::LinearSolve.LinearCache,::ExtendableSparseMatrix)
```

We can create a test problem and solve it with the `\` operator.
```@example
using ExtendableSparse # hide
A=fdrand(10,10,10,matrixtype=ExtendableSparseMatrix)
x=ones(1000)
b=A*x
y=A\b
sum(y)
```

The same problem can be solved by the tools available via `LinearSolve.jl`:
```@example
using ExtendableSparse # hide
using LinearSolve # hide
A=fdrand(10,10,10,matrixtype=ExtendableSparseMatrix)
x=ones(1000)
b=A*x
y=solve(LinearProblem(A,b),KLUFactorization()).u
sum(y)
```


Also, the iterative method interface works with the preconditioners defined in this package.
```@example
using ExtendableSparse # hide
using LinearSolve # hide
A=fdrand(10,10,10,matrixtype=ExtendableSparseMatrix)
x=ones(1000)
b=A*x
y=LinearSolve.solve(LinearProblem(A,b),IterativeSolversJL_CG(),Pl=ILU0Preconditioner(A)).u
sum(y)
```
