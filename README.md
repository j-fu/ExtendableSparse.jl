# ExtendableSparse.jl

[![Build Status](https://img.shields.io/travis/j-fu/ExtendableSparse.jl/master.svg?label=Linux+MacOSX+Windows)](https://travis-ci.org/j-fu/ExtendableSparse.jl)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://j-fu.github.io/ExtendableSparse.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://j-fu.github.io/ExtendableSparse.jl/dev)



Sparse matrix class with efficient successive insertion of entries.

Without an intermediate data structure, efficient successive insertion/update of entries in random order into a standard compressed colume storage structure appears to be not possible. The package introduces `ExtendableSparseMatrix`, a delegating wrapper around the Julia standard `SparseMatrixCSC` struct which contains an additional linked list based (but realised in vectors) temporary extension structure.

`ExtendableSparseMatrix` is aimed to work as a drop-in replacement to `SparseMatrixCSC` in finite element and finite volume codes.


