using ForwardDiff, LinearAlgebra
using ZeroSumGameSolve
using ReverseDiff
using CUDA
using Zygote: Zygote


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

export new_solve_static_unconstrained_zero_sum_GAN
function new_solve_static_unconstrained_zero_sum_GAN(guess, func, n_x, tol, max_iters, α, epoch)
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
    k = 0
    error = tol+1.0
    path = [x]
    while k<max_iters && error>tol
        # gradient
        grads = Zygote.gradient(x) do x
            func(x)
        end
        grads = grads[1]
        update_step_grad = zero_sum_gradient(grads, n_x)
        # hessian
        hessian = Zygote.hessian(x) do x
            func(x)
        end
        update_step_hess = zero_sum_hessian(hessian, n_x)
        # static_array_update = alpha(x, func, n_x)*inv(update_step_hess*(circle_theorem_regularize(update_step_hess'+update_step_hess))) * update_step_grad
        # static_array_update = α*inv(circle_theorem_regularize(update_step_hess*(update_step_hess'+update_step_hess))) * update_step_grad
        α_decayed = exp(-0.0001 * epoch)
        println("alpha: ", α_decayed)
        static_array_update = α_decayed * inv(circle_theorem_regularize(transpose(update_step_hess)*update_step_hess*(update_step_hess+transpose(update_step_hess)))) * transpose(update_step_hess)* update_step_grad
        # static_array_update = α* ((circle_theorem_regularize(transpose(update_step_hess)*update_step_hess*(update_step_hess+transpose(update_step_hess)))) \ (transpose(update_step_hess)* update_step_grad))
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
    # if k == max_iters
    #     println("Newton's method did not converge!")
    #     println("Error: ", error)
    # end
    return x, func(x), k, path
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

export new_regularization_solve_static_unconstrained_zero_sum
function new_regularization_solve_static_unconstrained_zero_sum(guess, func, n_x, tol, max_iters, α)
    x = guess
    k = 0
    error = tol+1.0
    path = [x]
    while k<max_iters && error>tol
        update_step_grad = zero_sum_gradient(x, func, n_x)
        update_step_hess = zero_sum_hessian(x, func, n_x)
        step = regularization(x, update_step_grad, update_step_hess)
        α = alpha(x, func, step, update_step_grad, update_step_hess)
        static_array_update = α*step
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

export new_reg_GAN
function new_reg_GAN(guess, func, n_x, tol, max_iters, epoch)
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
    k = 0
    error = tol+1.0
    path = [x]
    while k<max_iters && error>tol
        # gradient
        grads = Zygote.gradient(x) do x
            func(x)
        end
        grads = grads[1]
        update_step_grad = zero_sum_gradient(grads, n_x)
        # hessian
        hessian = Zygote.hessian(x) do x
            func(x)
        end
        update_step_hess = zero_sum_hessian(hessian, n_x)
        step = regularization(x, update_step_grad, update_step_hess)
        α = alpha(x, func, step, update_step_grad, update_step_hess)
        println("alpha: ", α)
        static_array_update = α*step
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
    # if k == max_iters
    #     println("Newton's method did not converge!")
    #     println("Error: ", error)
    # end
    return x, func(x), k, path
end