using Zygote
using LinearAlgebra
using StaticArrays
include("Dynamics.jl")
include("Objective.jl")
# B = [0 0 0; 0 0 0; 0 0 0; 1 0 0; 0 1 0; 0 0 1]
# C = [1 0 0 0 0 0; 0 1 0 0 0 0; 0 0 1 0 0 0]
# b_dyn = CW_Dynamics(0.1, 0.1, B, C)
# g_dyn = CW_Dynamics(0.1, 0.1, B, C)
# horizon = 5
# b_init = zeros(6,1)
# g_init = g_init = ones(6,1)
# b_controls = 0.1*ones(3*horizon)
# g_controls = 0.1*ones(3*horizon)
# controls = [b_controls; g_controls]

export orbital_stacked_gradients
function orbital_stacked_gradients(controls, b_dyn, g_dyn, b_init, g_init)
    grad_obj = controls -> objective(controls, b_dyn, g_dyn, b_init, g_init)
    # println(grad_obj(controls))
    gradient_orbital = Zygote.gradient(grad_obj, controls) # this gives tuple (gradient)
    # println(gradient_orbital)
    hessian_orbital = Zygote.hessian(grad_obj, controls)
    # println(hessian_orbital)
    return gradient_orbital[1], hessian_orbital
end

export convert_derivatives_to_zero_sum
function convert_derivatives_to_zero_sum(gradient, hessian)
    n_x = size(gradient)[1]÷2
    update_step_grad = [gradient[1:n_x]; -1.0*gradient[n_x+1:end]]
    update_step_hess = [hessian[1:n_x, 1:n_x] hessian[1:n_x, n_x+1:end]; -1.0*hessian[n_x+1:end, 1:n_x] -1.0*hessian[n_x+1:end, n_x+1:end]]
    return update_step_grad, update_step_hess
end

export orbital_zero_sum_derivatives
function orbital_zero_sum_derivatives(controls, b_dyn, g_dyn, b_init, g_init)
    gradient_orbital, hessian_orbital = orbital_stacked_gradients(controls, b_dyn, g_dyn, b_init, g_init)
    update_step_grad, update_step_hess = convert_derivatives_to_zero_sum(gradient_orbital, hessian_orbital)
    return update_step_grad, update_step_hess
end