using ForwardDiff, LinearAlgebra
export rosenbrock
function rosenbrock(x, y)
    # a = 1, b = 100
    return (1.0 - x)^2 + 100.0 * (y - x^2)^2
end

export circle_theorem_regularize
function circle_theorem_regularize(x)
    n = size(x)[1]
    ϵ = zeros(n, n)
    for i in 1:n
        c = x[i, i]
        r = norm(x[i, 1:n], 1) - abs(c)
        if c >= 0
            if c - r < 0
                ϵ[i, i] = r - c
            end
        else
            ϵ[i, i] = r + abs(c)
        end
    end
    return x + ϵ + 5.0*I(n)
end


# function for runge kutta integration with adaptive step size and tolerance for convergence_plot
export runge_kutta_adaptive
function runge_kutta_adaptive(func, guess, tol, total_iter, α=1e-4)
    x = guess
    k = 0
    eps = tol + 1.0
    path = [x]
    t = 0.0
    p = 0.0
    while eps > tol && k < total_iter
        k1 = func(x, t, p)
        k2 = func(x + 0.5*α*k1, t, p)
        k3 = func(x + 0.5*α*k2, t, p)
        k4 = func(x + α*k3, t, p)
        x_new = x + (α/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
        eps = norm(x_new - x)
        x = x_new
        push!(path, x)
        k += 1
    end
    if k == total_iter
        println("Runge-Kutta did not converge!. ODE Problem: ", func)
    end
    return x, func(x, t, p), k, path
end




# function convergence_plot(paths)
#     for path in paths
#         iters = size(path)[1]
#         ratios = zeros(iters)
#         converged_point = path[end]
#         for i in 1:iters
#             ratios[i] = norm(path[i] - converged_point)
#         end
# end






# function min_x_newton_step(func, guess, tol, total_iter, α=1.0)
#     x = guess
#     k = 0
#     eps = tol + 1.0
#     while eps > tol && k < total_iter
#         grad = ForwardDiff.gradient(func, x)
#         Hess = ForwardDiff.hessian(func, x)
#         Hess_reg = circle_theorem_regularize(Hess)
#         x_new = x - α*inv(Hess_reg) * grad
#         eps = norm(x_new - x)
#         x = x_new
#         k += 1
#     end
#     if k == total_iter
#         println("Newton's method did not converge!")
#     end
#     return x, func(x), ForwardDiff.gradient(func, x), ForwardDiff.hessian(func, x), k
# end

# point, value, grad, hess, iters = min_x_newton_step(rosenbrock, [-5.0, 10.0], 10e-6, 100000, 0.5)
# println("Point: ", point)
# println("Value: ", value)
# println("Gradient: ", grad)
# println("Hessian: ", hess)
# println("Iterations: ", iters)