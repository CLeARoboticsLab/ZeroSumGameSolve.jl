using ForwardDiff, LinearAlgebra
export rosenbrock
function rosenbrock(x)
    # a = 1, b = 100
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
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
    return x + ϵ + 10e-5*I(n)
end

function min_x_newton_step(func, guess, tol, total_iter, α=1.0)
    x = guess
    k = 0
    eps = tol + 1.0
    while eps > tol && k < total_iter
        grad = ForwardDiff.gradient(func, x)
        Hess = ForwardDiff.hessian(func, x)
        Hess_reg = circle_theorem_regularize(Hess)
        x_new = x - α*inv(Hess_reg) * grad
        eps = norm(x_new - x)
        x = x_new
        k += 1
    end
    if k == total_iter
        println("Newton's method did not converge!")
    end
    return x, func(x), ForwardDiff.gradient(func, x), ForwardDiff.hessian(func, x), k
end

point, value, grad, hess, iters = min_x_newton_step(rosenbrock, [-5.0, 10.0], 10e-6, 100000, 0.5)
println("Point: ", point)
println("Value: ", value)
println("Gradient: ", grad)
println("Hessian: ", hess)
println("Iterations: ", iters)