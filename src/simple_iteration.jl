"""
````        
simple!(u,A,b;
                 abstol::Real = zero(real(eltype(b))),
                 reltol::Real = sqrt(eps(real(eltype(b)))),
                 log=false,
                 maxiter=100,
                 P=nothing
                 ) -> solution, [history]
````

Simple iteration scheme ``u_{i+1}= u_i - P^{-1} (A u_i -b)`` with similar API as the methods in IterativeSolvers.jl.

"""
function simple!(u,
                 A,
                 b;
                 abstol::Real = zero(real(eltype(b))),
                 reltol::Real = sqrt(eps(real(eltype(b)))),
                 log = false,
                 maxiter = 100,
                 Pl = nothing)
    res = A * u - b # initial residual
    upd = similar(res)
    r0 = norm(res) # residual norm
    history = [r0]  # intialize history recording
    for i = 1:maxiter
        ldiv!(upd, Pl, res)
        u .-= upd # solve preconditioning system and update solution
        mul!(res, A, u)    # calculate residual
        res .-= b
        r = norm(res)       # residual norm
        push!(history, r)  # record in history 
        if (r / r0) < reltol || r < abstol    # check for tolerance
            break
        end
    end
    if log
        return u, Dict(:resnorm => history)
    else
        return u
    end
end

simple(A, b; kwargs...) = simple!(zeros(eltype(b), length(b)), A, b; kwargs...)
