include("Gradient_Orbital.jl")
include("OrbitalRegularization.jl")

# B = [0 0 0; 0 0 0; 0 0 0; 1 0 0; 0 1 0; 0 0 1]
# C = [1 0 0 0 0 0; 0 1 0 0 0 0; 0 0 1 0 0 0]
# b_dyn = CW_Dynamics(1.0854e-3, 10, B, C)   # CW_Dynamics(ω, δt, B, C)
# g_dyn = CW_Dynamics(1.0854e-3, 10, B, C)   # CW_Dynamics(ω, δt, B, C)
# horizon = 10
# b_init = [0.0 2.0 0.0 -0.00217 0.0 0.0]
# g_init = [0.0 1.0 0.0 -0.00109 0.0 0.0]
# controls = 0.1*ones(3*horizon*2)


export SecOND_orbital
function SecOND_orbital(controls, init_bandit_state, init_guard_state, tol, max_iters, α, ball_tol=1e-8)
    x = controls
    l1 = size(controls)[1] ÷ 2
    l2 = size(controls)[1] - l1
    k = 0
    error = tol+1.0
    path = [x]
    while k<max_iters && error>tol
        update_step_grad, update_step_hess = orbital_zero_sum_derivatives(x, b_dyn, g_dyn, init_bandit_state, init_guard_state)
        static_array_update, outside_ball = orbital_regularization_SecOND(update_step_grad, update_step_hess, ball_tol)
        update = zeros(size(static_array_update))
        for i in 1:size(static_array_update)[1]
            update[i] = static_array_update[i][1]
        end
        if outside_ball
            α_line = α #alpha_toy(x, func, update, update_step_grad, update_step_hess)
            update = α_line*update
        else
            update = α*update
        end
        x_new = x - update
        push!(path, x_new)
        error = norm(x_new - x)
        x = x_new
        k += 1
    end
    if k==max_iters
        println("SecOND did not converge!")
        println("Error: ", error)
    end
    return x, k, path
end

export DND_orbital
function DND_orbital(init_controls, init_bandit_state, init_guard_state, tol, max_iters, α, ball_tol=1e-8)
    x = init_controls
    l1 = size(controls)[1] ÷ 2
    l2 = size(controls)[1] - l1
    k = 0
    error = tol+1.0
    path = [x]
    while k<max_iters && error>tol
        update_step_grad, update_step_hess = orbital_zero_sum_derivatives(x, b_dyn, g_dyn, init_bandit_state, init_guard_state)
        static_array_update = orbital_regularization_dnd(update_step_grad, update_step_hess)
        update = zeros(size(static_array_update))
        for i in 1:size(static_array_update)[1]
            update[i] = static_array_update[i][1]
        end
        update = α*update
        x_new = x - update
        push!(path, x_new)
        error = norm(x_new - x)
        x = x_new
        k += 1
    end
    if k==max_iters
        println("DND did not converge!")
        println("Error: ", error)
    end
    return x, k, path 
end