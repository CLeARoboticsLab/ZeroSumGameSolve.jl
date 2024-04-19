using ZeroSumGameSolve
using Plots
# guess = [-15.0, -2.5]
guess = [-4.0, -9.0]
tol = 10e-8
max_iters = 100000
# α = 0.001
α = 0.1
n_x = 1
func = twodimexample
sol, value, iters_taken, path = mazumdar_two_timescale_approximation(guess, func, tol, max_iters, α)
println("Mazumdar done")
sol2, value2, iters_taken2, path2 = solve_static_unconstrained_zero_sum(guess, func, n_x, tol, max_iters, α)
println("Unconstrained Ours done")
sol3, value3, iters_taken3, path3 = new_solve_static_unconstrained_zero_sum(guess, func, n_x, tol, max_iters, α)
println("New Unconstrained Ours done")
sol4, value4, iters_taken4, path4 = simultaneous_gda(guess, func, tol, max_iters, α)
println("Simultaneous GDA done")
print(typeof(path))
println("Mazumdar done")
println("Solution: ", sol)
println("Value: ", value)
println("Iterations: ", iters_taken)
println("\n")
println("Unconstrained Ours done")
println("Solution: ", sol2)
println("Value: ", value2)
println("Iterations: ", iters_taken2)
println("\n")
println("New Unconstrained Ours done")
println("Solution: ", sol3)
println("Value: ", value3)
println("Iterations: ", iters_taken3)
println("\n")
println("Simultaneous GDA done")
println("Solution: ", sol4)
println("Value: ", value4)
println("Iterations: ", iters_taken4)
println("\n")

xlims = [-50, 20]
ylims = [-20, 20]
xs = range(xlims...; length = 100)
ys = range(ylims...; length = 100)
plt = contourf(xs, ys, twodimexample; color = :viridis)
plot!([x[1] for x in path], [x[2] for x in path], label="Mazumdar", title="Mazumdar Two Timescale Approximation", show=true)
plot!([x[1] for x in path2], [x[2] for x in path2], label="Unconstrained Ours", show=true)
plot!([x[1] for x in path3], [x[2] for x in path3], label="New Unconstrained Ours", show=true)
plot!([x[1] for x in path4], [x[2] for x in path4], label="Simultaneous GDA", show=true)