using ZeroSumGameSolve
using Plots
using Random

guess = [-4.358797680630015, 4.0]
# guess = [0.0, 0.0]
# guess = [-10., -2.5]
tol = 10e-7
max_iters = 10000000
α = 0.01
n_x = 1
func = twodimexample
sol, value, iters_taken, path = solve_static_constrained_zero_sum(guess, func, n_x, tol, max_iters, α, [-20, 20], [-20, 20.0])
println("Solution: ", sol)
println("Value: ", value)
println("Iterations: ", iters_taken)
xlims = [-20, 20]
ylims = [-20, 20]
xs = range(xlims...; length = 100)
ys = range(ylims...; length = 100)
# for i in size(path)[1]
#     x_plot[i] = path[i][1]
#     y_plot[i] = path[i][2]
# end
plt = contourf(xs, ys, twodimexample; color = :viridis)
plot!([x[1] for x in path], [x[2] for x in path], color=:red
    , xlims=xlims, ylims=ylims)
# xlims = [-20, 0]
# ylims = [-20, 5]
# plot!(Shape([xlims[1], xlims[1], xlims[2], xlims[2]], [ylims[1], ylims[2], ylims[2], ylims[1]]), opacity=0.2, color=:brown, label="Constraint")
