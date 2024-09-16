using ZeroSumGameSolve
using Plots
include("Objective.jl")
include("Dynamics.jl")
guess = [0.0, 0.0]
tol = 10e-5
ball_tol = 1e-2
max_iters = 50000
# α = 0.001
α = 0.01
n_x = 1
func = objective_pe
sol, value, iters_taken, path = DND(guess, func, n_x, tol, max_iters, α)
println("DND done")
sol2, value2, iters_taken2, path2 = SecOND(guess, func, n_x, tol, max_iters, α, ball_tol)
println("SecOND done")

pursuer_trajectory = evolve_agent(2.0, sol[1], [0.0, 0.0], 0:0.1:10, 0.1)
evader_trajectory = evolve_agent(1.0, sol[1], [1.0, 1.0], 0:0.1:10, 0.1)

#Plot trajectories on same graph
plot(pursuer_trajectory[1, :], pursuer_trajectory[2, :], label="Pursuer DND", title="Trajectories of Pursuer and Evader", xlabel="x", ylabel="y")
plot!(evader_trajectory[1, :], evader_trajectory[2, :], label="Evader DND")
pursuer_trajectory2 = evolve_agent(2.0, sol2[1], [0.0, 0.0], 0:0.1:10, 0.1)
evader_trajectory2 = evolve_agent(1.0, sol2[1], [1.0, 1.0], 0:0.1:10, 0.1)
plot!(pursuer_trajectory2[1, :], pursuer_trajectory2[2, :], label="Pursuer SecOND")
plot!(evader_trajectory2[1, :], evader_trajectory2[2, :], label="Evader SecOND")
