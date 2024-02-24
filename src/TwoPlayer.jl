using ForwardDiff, LinearAlgebra
using ZeroSumGameSolve


export solve_static_unconstrained_zero_sum
function solve_static_unconstrained_zero_sum(guess, func, n_x, tol, max_iters, α=0.001)
    x = guess
    k = 0
    error = tol+1.0
    path = [x]
    while k<max_iters && error>tol
        update_step_grad = zero_sum_gradient(x, func, n_x)
        update_step_hess = zero_sum_hessian(x, func, n_x)
        static_array_update = α*inv(update_step_hess) * update_step_grad
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

# export solve_zero_sum
# function solve_zero_sum(guess, tol, total_iter, α=1.0)
#     k = 0
#     error = tol+1.0
#     x = guess[1]
#     y = guess[2]
#     z = [x, y]
#     while k<total_iter && error>tol
#         # F = [∂j1/∂x1, ∂j1/∂x2] through ForwardDiff, then change sign for P2
#         F = [x_gradient_function(x, y), -1.0*y_gradient_function(x, y)]
#         # Calculate D i.e,e jacobian of F
#         H = hessian_function(x, y)
#         H_reg = circle_theorem_regularize(H)
#         new_z = z - α*inv(H_reg)*F
#         error = norm(new_z - z)
#         z = new_z
#         k += 1
#     end
#     if k == total_iter
#         println("Newton's method did not converge!")
#     end
#     val = objective_function(z[1], z[2])
#     return z, val, k
# end

# Point, J_1, iters = solve_zero_sum([20, -30], 10e-5, 100000, 0.1)
# println("P1_Strat: ", Point[1])
# println("P2_Strat: ", Point[2])
# println("Objective Value ", J_1)
# println("Iterations: ", iters)