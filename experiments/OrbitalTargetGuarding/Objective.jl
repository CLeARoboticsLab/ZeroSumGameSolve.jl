using Zygote
using LinearAlgebra
include("Dynamics.jl")


# Objective function - RETURNS Matrix{Float64} of shape {1, 1}
# Both bandit and guard controls are assumed to be an array of the shape {3, horizon}
export objective 
function objective(controls, bandit_dynamics::CW_Dynamics, guard_dynamics::CW_Dynamics, bandit_init_state, guard_init_state)
    dim = size(controls)[1]÷2
    bandit_controls = controls[1:dim]
    guard_controls = controls[dim+1:end]
    # println("bandit_controls: ", bandit_controls)
    # print(size(bandit_controls))
    # println("guard_controls: ", guard_controls)
    # print(size(guard_controls))
    horizon = size(bandit_controls)[1]÷3
    bandit_final_state = phi(bandit_dynamics.ω, bandit_dynamics.δt)^horizon * bandit_init_state + sum([phi(bandit_dynamics.ω, bandit_dynamics.δt)^(i+1) * bandit_dynamics.B * bandit_controls[3*(horizon - i - 1)-2 : 3*(horizon - i - 1)] for i in -1:horizon-2])
    # println("bandit_final_state: ", bandit_final_state)
    guard_final_state = phi(guard_dynamics.ω, guard_dynamics.δt)^horizon * guard_init_state + sum([phi(guard_dynamics.ω, guard_dynamics.δt)^(i+1) * guard_dynamics.B * guard_controls[3*(horizon - i - 1) - 2 : 3*(horizon - i - 1)] for i in -1:horizon-2])
    objective_value = 1.1*transpose(bandit_dynamics.C * bandit_final_state) * bandit_dynamics.C * bandit_final_state - transpose(guard_dynamics.C * (bandit_final_state - guard_final_state)) * guard_dynamics.C * (bandit_final_state - guard_final_state)
    return objective_value[1]
end








# #Objective function
# function objective(x, y, z)
#     return norm(x)^2 + norm(y)^2 + norm(z)^2
# end

# #Gradient of the objective function
# derivative_objective = y -> objective(x, y, z)
# x = [1.0, 2.0, 3.0]
# y = [14.0, 15.0, 16.0]
# z = [7.0, 8.0, 9.0]
# gradient_objective = Zygote.gradient(derivative_objective, y)
# hessian_objective = Zygote.hessian(derivative_objective, y)