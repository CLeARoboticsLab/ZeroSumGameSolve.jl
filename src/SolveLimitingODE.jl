using ZeroSumGameSolve
using DifferentialEquations
using Plots
# u0 = [-15.0, -2.5] # Mazumdar and simgd to go to "LNE" as in his paper, ours go to a place with correct sign
u0 = [-5.0, -9.0] # Mazumdar is weird here
# u0 = [-10.0, 3.0] # This goes to reverse LNE
# u0 = [8., 4.] # Nothing besides sim_gd converges
# tspan = (0.0, 10000.0)
# tspan_base = (0.0, 10.0)
# tspan_mazumdar = (0.0, 10.0)
# prob = ODEProblem(limiting_ode_derivative_regularized, u0, tspan)
# prob_unreg = ODEProblem(limiting_ode_derivative, u0, tspan)
# prob_base = ODEProblem(sim_gd, u0, tspan_base)
# prob_mazumdar = ODEProblem(mazumdar_ode, u0, tspan_mazumdar)
# sol = solve(prob)
# sol_unreg = solve(prob_unreg)
# sol_base = solve(prob_base)
# sol_mazumdar = solve(prob_mazumdar)
stepsize_rk4 = 1e-4
sol_ours, value_ours, iters_taken_ours, path_ours = runge_kutta_adaptive(limiting_ode_derivative, u0, 1e-8, 100000, 1000*stepsize_rk4)
println("Ours done")
sol_ours_reg, value_ours_reg, iters_taken_ours_reg, path_ours_reg = runge_kutta_adaptive(limiting_ode_derivative_regularized, u0, 1e-8, 100000, 1000*stepsize_rk4)
println("Ours reg done")
sol_base, value_base, iters_taken_base, path_base = runge_kutta_adaptive(sim_gd, u0, 1e-8, 100000, stepsize_rk4)
println("Base done")
sol_mazumdar, value_mazumdar, iters_taken_mazumdar, path_mazumdar = runge_kutta_adaptive(mazumdar_ode, u0, 1e-8, 100000, stepsize_rk4)
println("Mazumdar done")
sol_new, value_new, iters_taken_new, path_new = runge_kutta_adaptive(new_limiting_ode_derivative, u0, 1e-8, 100000, 1000*stepsize_rk4)

# println("\n")
# println("Ours")
# println("Solution: ", sol_ours)
# println("gradient: ", symbolic_zero_gradient(sol_ours[1], sol_ours[2]))
# println("second order matrix: ", symbolic_zero_hessian(sol_ours[1], sol_ours[2]))
# println("Iters: ", iters_taken_ours)

# println("\n")
# println("Ours Regularized")
# println("Solution: ", sol_ours_reg)
# println("gradient: ", symbolic_zero_gradient(sol_ours_reg[1], sol_ours_reg[2]))
# println("second order matrix: ", symbolic_zero_hessian(sol_ours_reg[1], sol_ours_reg[2]))
# println("Iters: ", iters_taken_ours_reg)

# println("\n")
# println("Base")
# println("Solution: ", sol_base)
# println("gradient: ", symbolic_zero_gradient(sol_base[1], sol_base[2]))
# println("second order matrix: ", symbolic_zero_hessian(sol_base[1], sol_base[2]))
# println("Iters: ", iters_taken_base)

println("\n")
println("Mazumdar")
println("Solution: ", sol_mazumdar)
println("gradient: ", symbolic_zero_gradient(sol_mazumdar[1], sol_mazumdar[2]))
println("second order matrix: ", symbolic_zero_hessian(sol_mazumdar[1], sol_mazumdar[2]))
println("Iters: ", iters_taken_mazumdar)

println("\n")
println("New")
println("Solution: ", sol_new)
println("gradient: ", symbolic_zero_gradient(sol_new[1], sol_new[2]))
println("second order matrix: ", symbolic_zero_hessian(sol_new[1], sol_new[2]))
println("Iters: ", iters_taken_new)





xlims = [-20, 20]
ylims = [-20, 20]
# println(sol[end])
xs = range(xlims...; length = 200)
ys = range(ylims...; length = 200)
plt = contourf(xs, ys, twodimexample; color = :viridis)
# plot!([x[1] for x in path_ours], [x[2] for x in path_ours], label="Ours", color=:black
#     , xlims=xlims, ylims=ylims, xlabel="x", ylabel="y"
#     , title="Limiting ODE Solution"
#     , legend=:bottomright)
# plot!([x[1] for x in path_ours_reg], [x[2] for x in path_ours_reg], label="Unregularized", color=:yellow
#     , xlims=xlims, ylims=ylims)
# plot!([x[1] for x in path_base], [x[2] for x in path_base], label="Sim GD", color=:red
#     , xlims=xlims, ylims=ylims)
plot!([x[1] for x in path_mazumdar], [x[2] for x in path_mazumdar], label="Mazumdar", color=:green
    , xlims=xlims, ylims=ylims)
plot!([x[1] for x in path_new], [x[2] for x in path_new], label="New", color=:blue
    , xlims=xlims, ylims=ylims)



# plot!(sol, vars=(1, 2), label="Ours", color=:black
#     , xlims=xlims, ylims=ylims, xlabel="x", ylabel="y"
#     , title="Limiting ODE Solution"
#     , legend=:bottomright, linestyle=:dash)
# plot!(sol_unreg, vars=(1, 2), label="Unregularized", color=:yellow
#     , xlims=xlims, ylims=ylims)
# plot!(sol_base, vars=(1, 2), label="Sim GD", color=:red
#     , xlims=xlims, ylims=ylims)
# plot!(sol_mazumdar, vars=(1, 2), label="Mazumdar", color=:green
#     , xlims=xlims, ylims=ylims)
