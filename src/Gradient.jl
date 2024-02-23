using ForwardDiff, StaticArrays


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

export zero_sum_gradient
function zero_sum_gradient(point, func, n_x)
    grad = gradient(point, func)
    update_step_grad = [grad[1:n_x], -1.0*grad[n_x+1:end]]
    update_step_grad = SVector{size(update_step_grad)...}(update_step_grad)
    return update_step_grad
end

export zero_sum_hessian
function zero_sum_hessian(point, func, n_x)
    hess = hessian(point, func)
    ∇_xx_reg = circle_theorem_regularize(hess[1:n_x, 1:n_x])
    ∇_yy_reg_neg = circle_theorem_regularize(-1.0*hess[n_x+1:end, n_x+1:end])
    update_step_hess = [∇_xx_reg hess[1:n_x, n_x+1:end]; -1.0*hess[n_x+1:end, 1:n_x] ∇_yy_reg_neg]
    update_step_hess = SMatrix{size(update_step_hess)...}(update_step_hess)
    return update_step_hess
end



# Some Basic Testing Functions used very early on in the repo development
function objf(x, y)
    return x^4+y^4
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