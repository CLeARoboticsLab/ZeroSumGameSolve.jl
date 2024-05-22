using ForwardDiff, LinearAlgebra
using ZeroSumGameSolve

export constrained_g_d
function constrained_g_d(guess, center, radius, func, n_x, tol, max_iters, α)
    x = guess
    k = 0
    error = tol+1.0
    path = [x]
    while k<max_iters && error>tol
        # println("k = ", k)
        update_step_grad = zero_sum_gradient(x, func, n_x)
        update_step_hess = zero_sum_true_hessian(x, func, n_x)
        static_array_update = regularization_g_d(update_step_grad, update_step_hess)
        update = zeros(size(static_array_update))
        for i in 1:size(static_array_update)[1]
            update[i] = static_array_update[i][1]
        end
        update = α*update
        if norm(x-center) < radius
            x_new = circle_projection(x - update, center, radius)
        else
            if norm(x-center) == radius
                println(x)
            end
            w = [update_step_grad[1][1], update_step_grad[2][1]]
            m = (transpose(update)*w/norm(w))*w
            m = m / norm(w)
            x_new = circle_projection(x - m, center, radius)
        end
        push!(path, x_new)
        error = norm(x_new - x)
        x = x_new
        k += 1
    end
    if k==max_iters
        println("constrained g_d did not converge!")
        println("Error: ", error)
    end
    return x, func(x[1], x[2]), k, path 
end