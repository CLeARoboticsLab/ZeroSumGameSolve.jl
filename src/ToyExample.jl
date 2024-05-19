using ZeroSumGameSolve
using Plots
# guess = [-15.0, -2.5]
guess = [-4.0, -9.0]
tol = 10e-8
ball_tol = 1e-5
max_iters = 100000
# α = 0.001
α = 0.01
n_x = 1
func = twodimexample
sol, value, iters_taken, path = mazumdar_two_timescale_approximation(guess, func, tol, max_iters, α)
println("Mazumdar done")
sol2, value2, iters_taken2, path2 = g_d(guess, func, n_x, tol, max_iters, α)
println("g_d done")
sol3, value3, iters_taken3, path3 = SecOND(guess, func, n_x, tol, max_iters, α, ball_tol)
println("SecOND done")
sol4, value4, iters_taken4, path4 = simultaneous_gda(guess, func, tol, max_iters, α)
println("Simultaneous GDA done")
sol5, value5, iters_taken5, path5 = new_regularization_solve_static_unconstrained_zero_sum(guess, func, n_x, tol, max_iters, α)
println("New Regularization done")
sol6, value6, iters_taken6, path6 = cesp(guess, func, tol, max_iters, α)
print(typeof(path))
println("Mazumdar done")
println("Solution: ", sol)
println("Value: ", value)
println("Iterations: ", iters_taken)
println("\n")
println("g_d done")
println("Solution: ", sol2)
println("Value: ", value2)
println("Iterations: ", iters_taken2)
println("\n")
println("SecOND done")
println("Solution: ", sol3)
println("Value: ", value3)
println("Iterations: ", iters_taken3)
println("\n")
println("Simultaneous GDA done")
println("Solution: ", sol4)
println("Value: ", value4)
println("Iterations: ", iters_taken4)
println("\n")
println("New Regularization done")
println("Solution: ", sol5)
println("Value: ", value5)
println("Iterations: ", iters_taken5)
println("\n")
println("CESP done")
println("Solution: ", sol6)
println("Value: ", value6)
println("Iterations: ", iters_taken6)
println("\n")

xlims = [-50, 20]
ylims = [-20, 20]
xs = range(xlims...; length = 100)
ys = range(ylims...; length = 100)
plt = contourf(xs, ys, twodimexample; color = :viridis)
plot!([x[1] for x in path], [x[2] for x in path], label="Mazumdar", show=true)
plot!([x[1] for x in path2], [x[2] for x in path2], label="g_d", show=true)
plot!([x[1] for x in path3], [x[2] for x in path3], label="SecOND", show=true)
plot!([x[1] for x in path4], [x[2] for x in path4], label="Simultaneous GDA", show=true)
plot!([x[1] for x in path5], [x[2] for x in path5], label="New Regularization", show=true)
plot!([x[1] for x in path6], [x[2] for x in path6], label="CESP", show=true)