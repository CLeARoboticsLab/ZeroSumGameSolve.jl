using ZeroSumGameSolve
using DifferentialEquations
using Plots

u0 = [-11.426651339751098]
u0[1] += 0.1
stepsize_rk4 = 1e-4
sol_ours, value_ours, iters_taken_ours, path_ours = runge_kutta_adaptive(test_ode_x, u0, 1e-8, 100000, stepsize_rk4)
println("Ours")
println("Solution: ", sol_ours)
println("gradient: ", symbolic_zero_gradient(sol_ours[1], 8.004295964291586))
println("second order matrix: ", symbolic_zero_hessian(sol_ours[1], 8.004295964291586))
println("Iters: ", iters_taken_ours)

xlims = [-20, 20]
ylims = [-20, 20]
# println(sol[end])
xs = range(xlims...; length = 200)
ys = range(ylims...; length = 200)
plt = contourf(xs, ys, twodimexample; color = :viridis)
plot!([x[1] for x in path_ours], [8.004295964291586 for x in path_ours], label="Ours", color=:black
    , xlims=xlims, ylims=ylims, xlabel="x", ylabel="y"
    , title="Limiting ODE Solution"
    , legend=:bottomright)
