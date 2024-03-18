using ZeroSumGameSolve
using DifferentialEquations
using Plots
# u0 = [-15.0, -2.5]
u0 = [-4.0, -9.0]
tspan = (0.0, 10000.0)
tspan_base = (0.0, 10.0)
prob = ODEProblem(limiting_ode_derivative_regularized, u0, tspan)
prob_unreg = ODEProblem(limiting_ode_derivative, u0, tspan)
prob_base = ODEProblem(sim_gd, u0, tspan_base)
sol = solve(prob)
sol_unreg = solve(prob_unreg)
sol_base = solve(prob_base)
xlims = [-20, 20]
ylims = [-20, 20]
println(sol[end])
xs = range(xlims...; length = 100)
ys = range(ylims...; length = 100)
plt = contourf(xs, ys, twodimexample; color = :viridis)
plot!(sol, vars=(1, 2), label="Ours", color=:black
    , xlims=xlims, ylims=ylims, xlabel="x", ylabel="y"
    , title="Limiting ODE Solution"
    , legend=:bottomright, linestyle=:dash)
plot!(sol_unreg, vars=(1, 2), label="Unregularized", color=:yellow
    , xlims=xlims, ylims=ylims)
plot!(sol_base, vars=(1, 2), label="Sim GD", color=:red
    , xlims=xlims, ylims=ylims)
