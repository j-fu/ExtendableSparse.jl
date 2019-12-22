# Changes
## dev
- Tried to track down the source from which I learned the linked list based struct in order
  to document this. Ended up with SPARSEKIT of Y.Saad, however I believe this 
  already was in SPARSEPAK by Chu,George,Liu.
- Internal rename of SparseMatrixExtension to SparseMatrixLNK. 

## v0.2 Dec 2019
- more interface methods delegating to csc, in particular mul! and ldiv!
- lazy creation of extendable part: don't create idle memory
- nicer constructors
  
## V0.1, July 2019
- Initial release

