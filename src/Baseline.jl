using ZeroSumGameSolve
using LinearAlgebra
using Plots
using Optimisers: Optimisers

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

export GAN_mazumdar_two_timescale_approximation
function GAN_mazumdar_two_timescale_approximation!(guess, func, n_x, tol, max_iters, epoch, gamma_1=0.004, gamma_2=0.005, xi_1=1e-4, xi_2=1e-3; xy_optimizer_setup, v_optimizer_setup)
    function zero_sum_gradient(grads, n_x)
        update_step_grad = vcat(grads[1:n_x], -1.0*grads[n_x+1:end])
        update_step_grad = SVector{size(update_step_grad)...}(update_step_grad)
        return update_step_grad
    end
    function zero_sum_hessian(hess, n_x)
        ∇_xx_reg = circle_theorem_regularize(hess[1:n_x, 1:n_x])
        ∇_yy_reg_neg = circle_theorem_regularize(-1.0*hess[n_x+1:end, n_x+1:end])
        update_step_hess = [∇_xx_reg hess[1:n_x, n_x+1:end]; -1.0*hess[n_x+1:end, 1:n_x] ∇_yy_reg_neg]
        update_step_hess = SMatrix{size(update_step_hess)...}(update_step_hess)
        return update_step_hess
    end
    x = guess
    v = zeros(size(x))
    k = 0
    error = tol+1.0
    path = [x]
    while k<max_iters && error>tol
        # w = zero_sum_gradient(x, twodimexample, n_x)
        # J = zero_sum_true_hessian(x, twodimexample, n_x)
        grads = Zygote.gradient(x) do x
            func(x)
        end
        grads = grads[1]
        w = zero_sum_gradient(grads, n_x)
        # hessian
        hessian = Zygote.hessian(x) do x
            func(x)
        end
        J = zero_sum_hessian(hessian, n_x)
        J_t = transpose(J)
        λ = xi_1*(1-exp(-1.0*(LinearAlgebra.norm(w)^2)))*LinearAlgebra.I(length(guess))
        update = exp(-xi_2*(LinearAlgebra.norm(J_t*v)^2))J_t*v
        w_update = zeros(size(w))
        for i in 1:size(w)[1]
            w_update[i] = w[i][1]
        end
        arr_update = zeros(size(update))
        for i in 1:size(update)[1]
            arr_update[i] = update[i][1]
        end
        xy_optimizer_setup, x_new = Optimisers.update!(xy_optimizer_setup, x, w_update + arr_update)
        # x_new = x - gamma_1*(w_update + arr_update)
        update = -J_t*w
        arr_update = zeros(size(update))
        for i in 1:size(update)[1]
            arr_update[i] = update[i][1]
        end
        v_optimizer_setup, v = Optimisers.update!(v_optimizer_setup, v, J_t*J*v + λ*v +arr_update)
        # v = v - 0.005*(J_t*J*v + λ*v +arr_update)
        error = norm(x_new - x)
        x = x_new
        push!(path, x)
        k = k + 1
    end
    return x, func(x), k, path
end
