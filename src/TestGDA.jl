using ZeroSumGameSolve
using Plots
using Random
using StatsPlots


function test()
tol = 1e-7
ball_tol = 1e-2
max_iters = 50000
α = 0.0001
n_x = 1
func = twodimexample
max_val_x = 3
min_val_x = -3
n = 1

max_val_y = 3
min_val_y = -3
# guess = [-15., 5.]
guess = [-11., -3.2]
points = []
for i in 1:n
    x = rand()*(max_val_x - min_val_x) + min_val_x
    y = rand()*(max_val_y - min_val_y) + min_val_y
    push!(points, [x, y])
end
number_points = n
simultaneous_gda_iters = []
xlims = [-50, 20]
ylims = [-20, 20]
xs = range(xlims...; length = 100)
ys = range(ylims...; length = 100)
plt = contourf(xs, ys, twodimexample; color = :viridis)
k_con = 0
k_ncov = 0
for point in 1:number_points
    sol, value, iters_taken, path, converged = toy_cesp(guess, func, n_x, tol, max_iters, α)
    println(sol)
    println(zero_sum_gradient(sol, func, 1))
    println(zero_sum_true_hessian(sol, func, 1))
    if converged == false
        k_ncov += 1
        plot!([x[1] for x in path], [x[2] for x in path], color = :red, label="CESP", show=true)
    else
        k_con += 1
        plot!([x[1] for x in path], [x[2] for x in path], color = :black, label="CESP", show=true)
    end
    println("Iterations: ", iters_taken)
end
println("Converged: ", k_con)
println("Not Converged: ", k_ncov)
end

test()