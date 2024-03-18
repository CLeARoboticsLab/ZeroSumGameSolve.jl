using ZeroSumGameSolve

export limiting_ode_derivative
function limiting_ode_derivative(point, p, t)
    zero_sum_hess = zero_sum_hessian(point, twodimexample, 1)
    zero_sum_grad = zero_sum_gradient(point, twodimexample, 1)
    term = -1.0*inv(zero_sum_hess*(zero_sum_hess+zero_sum_hess')) * zero_sum_grad
    du = zeros(2)
    du[1] = term[1][1]
    du[2] = term[2][1]
    return du
end

export limiting_ode_derivative_regularized
function limiting_ode_derivative_regularized(point, p, t)
    zero_sum_hess = zero_sum_hessian(point, twodimexample, 1)
    zero_sum_grad = zero_sum_gradient(point, twodimexample, 1)
    term = -1.0*inv(circle_theorem_regularize(zero_sum_hess*(zero_sum_hess+zero_sum_hess'))) * zero_sum_grad
    du = zeros(2)
    du[1] = term[1][1]
    du[2] = term[2][1]
    return du
end

export sim_gd
function sim_gd(point, p, t)
    du = zeros(2)
    term = zero_sum_gradient(point, twodimexample, 1)
    du[1] = term[1][1]
    du[2] = term[2][1]
    return du
end
