using ForwardDiff
using MultiFloats
f64(x::ForwardDiff.Dual{T}) where T=Float64(ForwardDiff.value(x))
f64(x::Number)=Float64(x)
const Dual64=ForwardDiff.Dual{Float64,Float64,1}


function test_lu1(T,k,l,m; lufac=ExtendableSparse.LUFactorization())
    Acsc=fdrand(k,l,m,rand=()->1,matrixtype=SparseMatrixCSC)
    b=rand(k*l*m)
    LUcsc=lu(Acsc)
    x1csc=LUcsc\b
    for i=1:k*l*m
        Acsc[i,i]+=1.0
    end
    LUcsc=lu!(LUcsc,Acsc)
    x2csc=LUcsc\b

    Aext=fdrand(T,k,l,m,rand=()->1,matrixtype=ExtendableSparseMatrix)
    lu!(lufac,Aext)
    x1ext=lufac\b
    for i=1:k*l*m
        Aext[i,i]+=1.0
    end
    update!(lufac)
    x2ext=lufac\b
    x1csc≈f64.(x1ext) && x2csc ≈ f64.(x2ext)
end

function test_lu2(T,k,l,m;lufac=ExtendableSparse.LUFactorization())
    Aext=fdrand(T,k,l,m,rand=()->1,matrixtype=ExtendableSparseMatrix)
    b=rand(k*l*m)
    lu!(lufac,Aext)
    x1ext=lufac\b
    for i=4:k*l*m-3
        Aext[i,i+3]-=1.0e-4
        Aext[i-3,i]-=1.0e-4
    end
    lufac=lu!(lufac,Aext)
    x2ext=lufac\b
    all(x1ext.< x2ext)
end
