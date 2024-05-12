using ForwardDiff, StaticArrays


export gradient
function gradient(point, func)
    a = SVector{size(point)...}(point)
    return ReverseDiff.gradient(x->func(x...), a)
end


## Gives out
## J = [∇_xx,   ∇_xy]
##     [∇_yx,   ∇_yy]
export hessian
function hessian(point, func)
    a = SVector{size(point)...}(point)
    return ReverseDiff.hessian(x->func(x...), a)
end


## Takes the form
## w = [∇_x; -∇_y]
export zero_sum_gradient
function zero_sum_gradient(point, func, n_x)
    grad = gradient(point, func)
    update_step_grad = [grad[1:n_x], -1.0*grad[n_x+1:end]]
    update_step_grad = SVector{size(update_step_grad)...}(update_step_grad)
    return update_step_grad
end


## Fundamental matrix of the paper.
## J = [∇_xx,   ∇_xy]
##     [-∇_yx, -∇_yy]
export zero_sum_true_hessian
function zero_sum_true_hessian(point, func, n_x)
    hess = hessian(point, func)
    ∇_xx = hess[1:n_x, 1:n_x]
    ∇_xy = hess[1:n_x, n_x+1:end]
    ∇_yx = -1.0*hess[n_x+1:end, 1:n_x]
    ∇_yy = -1.0*hess[n_x+1:end, n_x+1:end]
    update_step_hess = [∇_xx ∇_xy; ∇_yx ∇_yy]
    update_step_hess = SMatrix{size(update_step_hess)...}(update_step_hess)
end


## Regularized version of the fundamental matrix of the paper.
## J + E = [∇_xx + E₁,        ∇_xy]
##         [-∇_yx,      -∇_yy + E₂]
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
export objf
function objf(x, y)
    return x+y^4
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



#∇_x = exp(-0.01*x^2 - 0.01*y^2)*(-0.0018*x^5 + (0.34 - 0.012* y)*x^3 + y^2 - 0.02* x^2 * y^2 + x *(2 + 1.2* y - 0.02 *y^2 - 0.005* y^4))
#∇_y = exp(-0.01*x^2 - 0.01*y^2)*(2*y - 0.0018*x^4 * y + 0.98 * y^3 - 0.005 * y^5 + x^2 * (0.6 - 0.02 * y - 0.012 * y^2) + x * (2 *y - 0.02 *y^3))


export symbolic_zero_gradient
function symbolic_zero_gradient(x, y)
    expterm = exp(-0.01*x^2 - 0.01*y^2)
    func = twodimexample(x, y)
    ∇_x = -0.02*x*func +expterm*(0.36*x^3+1.2*y*x+y^2+2*x)
    ∇_y = -0.02*y*func + expterm*(0.6*x^2+2*y+y^3+2*x*y)
    return @SVector [∇_x, -1.0*∇_y]
end

export symbolic_zero_hessian
function symbolic_zero_hessian(x, y)
    func = twodimexample(x, y)
    expterm = exp(-0.01*x^2 - 0.01*y^2)
    ∇_x = -0.02*x*func +expterm*(0.36*x^3+1.2*y*x+y^2+2*x)
    ∇_y = -0.02*y*func + expterm*(0.6*x^2+2*y+y^3+2*x*y)    
    ∇_xx = -0.02*func - 0.02*x*∇_x - 0.02*x*expterm*(0.36*x^3+1.2*y*x+y^2+2*x) + expterm*(1.08*x^2+1.2*y+2)
    ∇_yy = -0.02*func - 0.02*y*∇_y - 0.02*y*expterm*(0.6*x^2+2*y+y^3+2*x*y) + expterm*(2.0+3*y^2+2*x)
    ∇_xy = -0.02*x*∇_y - 0.02*y*expterm*(0.36*x^3+1.2*y*x+y^2+2*x) + expterm*(1.2*x+2*y)
    ∇_yx = -0.02*y*∇_x - 0.02*x*expterm*(0.6*x^2+2*y+y^3+2*x*y) + expterm*(1.2*x+2*y)
    J = @SMatrix [∇_xx ∇_yx; -1.0*∇_xy -1.0*∇_yy]
    return J
end

export symbolic_gradient_hessian
function symbolic_gradient_hessian(x, y)
    func = twodimexample(x, y)
    expterm = exp(-0.01*x^2 - 0.01*y^2)
    ∇_x = -0.02*x*func +expterm*(0.36*x^3+1.2*y*x+y^2+2*x)
    ∇_y = -0.02*y*func + expterm*(0.6*x^2+2*y+y^3+2*x*y)    
    ∇_xx = -0.02*func - 0.02*x*∇_x - 0.02*x*expterm*(0.36*x^3+1.2*y*x+y^2+2*x) + expterm*(1.08*x^2+1.2*y+2)
    ∇_yy = -0.02*func - 0.02*y*∇_y - 0.02*y*expterm*(0.6*x^2+2*y+y^3+2*x*y) + expterm*(2.0+3*y^2+2*x)
    ∇_xy = -0.02*x*∇_y - 0.02*y*expterm*(0.36*x^3+1.2*y*x+y^2+2*x) + expterm*(1.2*x+2*y)
    ∇_yx = -0.02*y*∇_x - 0.02*x*expterm*(0.6*x^2+2*y+y^3+2*x*y) + expterm*(1.2*x+2*y)
    ∇_xx_reg = circle_theorem_regularize([∇_xx])
    ∇_yy_reg = circle_theorem_regularize([-1.0*∇_yy])
    w = [∇_x, -1.0*∇_y]
    J = [∇_xx_reg ∇_yx; -1.0*∇_xy ∇_yy_reg]
    return w, J
end