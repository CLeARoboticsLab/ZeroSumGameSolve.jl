using Zygote
using LinearAlgebra
using StaticArrays
using Plots
include("Dynamics.jl")
include("Objective.jl")
include("Gradient_Orbital.jl")
include("OrbitalRegularization.jl")
include("Solvers_Orbital.jl")
B = [0 0 0; 0 0 0; 0 0 0; 1 0 0; 0 1 0; 0 0 1]
C = [1 0 0 0 0 0; 0 1 0 0 0 0; 0 0 1 0 0 0]
b_dyn = CW_Dynamics(1.0854e-3, 10, B, C)   # CW_Dynamics(ω, δt, B, C)
g_dyn = CW_Dynamics(1.0854e-3, 10, B, C)   # CW_Dynamics(ω, δt, B, C)
horizon = 10
b_init = [0.0; 2.0; 0.0; -0.00217; 0.0; 0.0]
g_init = [0.0; 1.0; 0.0; -0.00109; 0.0; 0.0]
controls = 0.01*ones(3*horizon*2)
sol, k, path = DND_orbital(controls, b_init, g_init, 1e-8, 1000, 0.1)
bandit_cood_sols, guard_cood_sols = unroll_trajectory(sol, b_dyn, g_dyn, b_init, g_init)

bandit_x = [bandit_cood_sols[i][1] for i in 1:horizon+1]
bandit_y = [bandit_cood_sols[i][2] for i in 1:horizon+1]
bandit_z = [bandit_cood_sols[i][3] for i in 1:horizon+1]
guard_x = [guard_cood_sols[i][1] for i in 1:horizon+1]
guard_y = [guard_cood_sols[i][2] for i in 1:horizon+1]
guard_z = [guard_cood_sols[i][3] for i in 1:horizon+1]

plot(bandit_x, bandit_y, bandit_z, label="Bandit", color="red", title="Orbital Target Guarding", xlabel="x", ylabel="y", zlabel="z")
plot!(guard_x, guard_y, guard_z, label="Guard", color="green")