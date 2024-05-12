using ForwardDiff, LinearAlgebra
using ZeroSumGameSolve


export solve_static_unconstrained_zero_sum
function solve_static_unconstrained_zero_sum(guess, func, n_x, tol, max_iters, α)
    x = guess
    k = 0
    error = tol+1.0
    path = [x]
    while k<max_iters && error>tol
        update_step_grad = zero_sum_gradient(x, func, n_x)
        update_step_hess = zero_sum_hessian(x, func, n_x)
        # static_array_update = alpha(x, func, n_x)*inv(update_step_hess*(circle_theorem_regularize(update_step_hess'+update_step_hess))) * update_step_grad
        static_array_update = α*inv(circle_theorem_regularize(update_step_hess*(update_step_hess'+update_step_hess))) * update_step_grad
        update = zeros(size(static_array_update))
        for i in 1:size(static_array_update)[1]
            update[i] = static_array_update[i][1]
        end
        x_new = x - update
        push!(path, x_new)
        error = norm(x_new - x)
        x = x_new
        k += 1
    end
    if k == max_iters
        println("Newton's method did not converge!")
        println("Error: ", error)
    end
    return x, func(x[1], x[2]), k, path
    
end

export new_solve_static_unconstrained_zero_sum
function new_solve_static_unconstrained_zero_sum(guess, func, n_x, tol, max_iters, α)
    x = guess
    k = 0
    error = tol+1.0
    path = [x]
    while k<max_iters && error>tol
        update_step_grad = zero_sum_gradient(x, func, n_x)
        update_step_hess = zero_sum_hessian(x, func, n_x)
        # static_array_update = alpha(x, func, n_x)*inv(update_step_hess*(circle_theorem_regularize(update_step_hess'+update_step_hess))) * update_step_grad
        # static_array_update = α*inv(circle_theorem_regularize(update_step_hess*(update_step_hess'+update_step_hess))) * update_step_grad
        static_array_update = α*inv(circle_theorem_regularize(transpose(update_step_hess)*update_step_hess*(update_step_hess+transpose(update_step_hess)))) * transpose(update_step_hess)* update_step_grad
        update = zeros(size(static_array_update))
        for i in 1:size(static_array_update)[1]
            update[i] = static_array_update[i][1]
        end
        x_new = x - update
        push!(path, x_new)
        error = norm(x_new - x)
        x = x_new
        k += 1
    end
    if k == max_iters
        println("Newton's method did not converge!")
        println("Error: ", error)
    end
    return x, func(x[1], x[2]), k, path
end

export new_regularization_solve_static_unconstrained_zero_sum
function new_regularization_solve_static_unconstrained_zero_sum(guess, func, n_x, tol, max_iters, α)
    x = guess
    k = 0
    error = tol+1.0
    path = [x]
    while k<max_iters && error>tol
        static_array_update = α*regularization(x, func, n_x)
        update = zeros(size(static_array_update))
        for i in 1:size(static_array_update)[1]
            update[i] = static_array_update[i][1]
        end
        x_new = x - update
        push!(path, x_new)
        error = norm(x_new - x)
        x = x_new
        k += 1
    end
    if k == max_iters
        println("Newton's method did not converge!")
        println("Error: ", error)
    end
    return x, func(x[1], x[2]), k, path
end

export solve_static_constrained_zero_sum
function solve_static_constrained_zero_sum(guess, func, n_x, tol, max_iters, α, xlims, ylims)
    x = guess
    k = 0
    error = tol+1.0
    path = [x]
    while k<max_iters && error>tol
        update_step_grad = zero_sum_gradient(x, func, n_x)
        update_step_hess = zero_sum_hessian(x, func, n_x)
        # static_array_update = alpha(x, func, n_x)*inv(update_step_hess*(circle_theorem_regularize(update_step_hess'+update_step_hess))) * update_step_grad
        static_array_update = α*inv(circle_theorem_regularize(update_step_hess*(update_step_hess'+update_step_hess))) * update_step_grad
        update = zeros(size(static_array_update))
        for i in 1:size(static_array_update)[1]
            update[i] = static_array_update[i][1]
        end
        x_new = x - update
        x_new = rectangle_projection(x_new, xlims, ylims)
        push!(path, x_new)
        error = norm(x_new - x)
        x = x_new
        k += 1
    end
    if k == max_iters
        println("Newton's method did not converge!")
        println("Error: ", error)
    end
    return x, func(x[1], x[2]), k, path
    
end