using ForwardDiff, LinearAlgebra
using ZeroSumGameSolve

export solver_qre
function solver_qre(guess, func, n_x, tol, max_iters, α)
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
        w = [update_step_grad[1][1], update_step_grad[2][1]]
        m = (transpose(update)*w/norm(w))*w
        m = m / norm(w)
        x_new = simplex_projection(x-α*m)
        push!(path, x_new)
        error = norm(x_new - x)
        println(x)
        x = x_new
        k += 1
    end
    println("iters: ", k)
    if k==max_iters
        println("constrained g_d did not converge!")
        println("Error: ", error)
    end
    return x, func(x[1], x[2]), k, path 
end

function simplex_projection(x)
    if x[2] - x[1] >= 1
        simplex_point = [1e-3, 1-1e-3]
    elseif x[2] - x[1] <= -1
        simplex_point = [1-1e-3, 1e-3]
    else
        b = 0.5*(x[1] + x[2] - 1)
        simplex_point = [x[1]- b, x[2] - b]
    end
    return simplex_point
end
