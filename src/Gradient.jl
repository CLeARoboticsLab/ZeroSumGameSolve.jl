using ForwardDiff, StaticArrays

function objf(x, y)
    return x^4+y^4
end

export gradient
function gradient(point, func)
    a = SVector{size(point)...}(point)
    return ForwardDiff.gradient(x->func(x...), a)
end

export hessian
function hessian(point, func)
    a = SVector{size(point)...}(point)
    return ForwardDiff.hessian(x->func(x...), a)
end

function test_grad(guess::Vector{Float64})
    a = SVector{2}(guess)
    # println(a)
    # println(typeof(a))
    return ForwardDiff.gradient(x->objf(x...), a)
end

function test_hessian(guess)
    a = SVector{2}(guess)
    return ForwardDiff.hessian(x->objf(x...), a)
end