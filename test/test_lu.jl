using ForwardDiff
using MultiFloats
f64(x::ForwardDiff.Dual{T}) where T=Float64(ForwardDiff.value(x))
f64(x::Number)=Float64(x)
const Dual64=ForwardDiff.Dual{Float64,Float64,1}


function test_lu1(T,k,l,m; lufac=ExtendableSparse.LUFactorization())
    A=fdrand(k,l,m,rand=()->1,matrixtype=ExtendableSparseMatrix)
    b=rand(k*l*m)
    x1=A\b
    for i=1:k*l*m
        A[i,i]+=1.0
    end
    x2=A\b
    
    Aext=fdrand(T,k,l,m,rand=()->1,matrixtype=ExtendableSparseMatrix)
    lu!(lufac,Aext)
    a1=deepcopy(Aext.cscmatrix)
    x1ext=lufac\b

    for i=1:k*l*m
        Aext[i,i]+=1.0
    end
    update!(lufac)
    a2=deepcopy(Aext.cscmatrix)
    x2ext=lufac\b

    atol=100*sqrt(f64(max(eps(T), eps(Float64))))
    isapprox(norm(x1-f64.(x1ext))/norm(x1),0;atol) &&    isapprox(norm(x2-f64.(x2ext))/norm(x2),0;atol)
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
