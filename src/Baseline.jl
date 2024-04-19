using ZeroSumGameSolve
using LinearAlgebra
using Plots

export mazumdar_ode
function mazumdar_ode(point, p, t)
    # w = zero_sum_gradient(point, twodimexample, 1)
    # J = zero_sum_true_hessian(point, twodimexample, 1)
    w = symbolic_zero_gradient(point[1], point[2])
    J = symbolic_zero_hessian(point[1], point[2])
    J_t = transpose(J)
    λ = 0.0001*(1-exp(-1.0*((norm(w))^2)))*LinearAlgebra.I(2)
    # typeof(v) = SVector{2, Vector{Float64}}
    v = J_t*inv(J_t*J+λ)*J_t*w
    # damping as descibed by mazumdar
    g = exp(-0.0001*((norm(v))^2))
    # term = -1.0*(w+g*v)
    term = (w+g*v)
    du = zeros(2)
    du[1] = term[1][1]
    du[2] = term[2][1]
    return du
end

export mazumdar_two_timescale_approximation
function mazumdar_two_timescale_approximation(guess, func, tol, max_iters, α)
    x = guess
    v = [1000., -1000.]
    k = 0
    error = tol+1.0
    path = [x]
    while k<max_iters && error>tol
        # w = zero_sum_gradient(x, twodimexample, n_x)
        # J = zero_sum_true_hessian(x, twodimexample, n_x)
        w = symbolic_zero_gradient(x[1], x[2])
        J = symbolic_zero_hessian(x[1], x[2])
        J_t = transpose(J)
        λ = 0.0001*(1-exp(-1.0*(LinearAlgebra.norm(w)^2)))*LinearAlgebra.I(2)
        update = exp(-0.0001*(LinearAlgebra.norm(J_t*v)^2))J_t*v
        x_new = x - 0.004*([w[1][1], w[2][1]] + [update[1], update[2]])
        update = -J_t*w
        v = v - 0.005*(J_t*J*v + λ*v +[update[1][1], update[2][1]])
        error = norm(x_new - x)
        x = x_new
        push!(path, x)
        k = k + 1
    end
    return x, func(x[1], x[2]), k, path
end

export simultaneous_gda
function simultaneous_gda(guess, func, tol, max_iters, α)
    x = guess
    k = 0
    error = tol+1.0
    path = [x]
    while k<max_iters && error>tol
        w = symbolic_zero_gradient(x[1], x[2])
        update = -1.0*w
        x_new = x + α*([update[1][1], update[2][1]])
        error = norm(x_new - x)
        x = x_new
        push!(path, x)
        k = k + 1
    end
    return x, func(x[1], x[2]), k, path
end