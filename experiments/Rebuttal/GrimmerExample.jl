using ZeroSumGameSolve
using Plots
guess = [4.0, 4.0]
tol = 10e-11
ball_tol = 1e-2
max_iters = 500000
# α = 0.001
α = 0.01
n_x = 1
func = grimmer
sol2, value2, iters_taken2, path2 = g_d(guess, func, n_x, tol, max_iters, α)
println("g_d done")
println("Solution: ", sol2)
println("Value: ", value2)
println("Iterations: ", iters_taken2)
xlims = [-4, 4]
ylims = [-4, 4]
xs = range(xlims...; length = 1000)
ys = range(ylims...; length = 1000)
plt = contourf(xs, ys, grimmer; color = :viridis)
nashpt = [0, 0]
plot!([x[1] for x in path2], [x[2] for x in path2], label="\$\\texttt{DND}\$", show=true)
scatter!([nashpt[1]], [nashpt[2]], color=:green, label="Local Nash Equilibrium", markershape=:star5, markersize=12)
