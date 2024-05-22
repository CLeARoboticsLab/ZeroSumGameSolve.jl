using ZeroSumGameSolve
using Plots
using Random
using StatsPlots
using Colors


function example1()
tol = 10e-5
ball_tol = 1e-2
max_iters = 15000
α = 0.0001
n_x = 1
func = twodimexample

max_val_x = 10
min_val_x = -10
n = 10000

max_val_y = 10
min_val_y = -10

points = []
for i in 1:n
    x = rand()*(max_val_x - min_val_x) + min_val_x
    y = rand()*(max_val_y - min_val_y) + min_val_y
    push!(points, [x, y])
end
second_failed = 0
number_points = n
g_d_iters = []
mazumdar_two_timescale_approximation_iters = []
SecOND_iters = []
simultaneous_gda_iters = []
diff_iters_mazumdar = []
diff_iters_g_d = []
diff_iters_simultaneous_gda = []
# new_regularization_solve_static_unconstrained_zero_sum_iters = []
cesp_iters = []
for point in 1:number_points
    guess = points[point]
    sol, value, iters_taken, path, converged = toy_mazumdar(guess, func, n_x, tol, max_iters, α)
    println("Mazumdar iterations: ", iters_taken)
    push!(mazumdar_two_timescale_approximation_iters, iters_taken)
    sol2, value2, iters_taken2, path2, converged2 = toy_g_d(guess, func, n_x, tol, max_iters, α)
    println("g_d iterations: ", iters_taken2)
    push!(g_d_iters, iters_taken2)
    sol3, value3, iters_taken3, path3, converged3 = toy_SecOND(guess, func, n_x, tol, max_iters, α, ball_tol)
    println("SecOND iterations: ", iters_taken3)
    push!(SecOND_iters, iters_taken3)
    sol4, value4, iters_taken4, path4, converged4 = toy_simultaneous_gda(guess, func, tol, max_iters, α)
    println("Simultaneous GDA iterations: ", iters_taken4)
    push!(simultaneous_gda_iters, iters_taken4)
    # sol5, value5, iters_taken5, path5 = new_regularization_solve_static_unconstrained_zero_sum(guess, func, n_x, tol, max_iters, α)
    # push!(new_regularization_solve_static_unconstrained_zero_sum_iters, iters_taken5)
    # sol6, value6, iters_taken6, path6, converged6 = toy_cesp(guess, func, tol, max_iters, α)
    # println("CESP iterations: ", iters_taken6)
    # push!(cesp_iters, iters_taken6)
    if !converged3
        second_failed += 1
    end
    if converged && converged3
        push!(diff_iters_mazumdar, iters_taken3 - iters_taken)
    end
    if converged2 && converged3
        push!(diff_iters_g_d, iters_taken3 - iters_taken2)
    end
    if converged4 && converged3
        push!(diff_iters_simultaneous_gda, iters_taken3 - iters_taken4)
    end
end

# diff_iters_mazumdar = SecOND_iters .- mazumdar_two_timescale_approximation_iters
# diff_iters_g_d = SecOND_iters .- g_d_iters
# diff_iters_simultaneous_gda = SecOND_iters .- simultaneous_gda_iters
# diff_iters_new_regularization = SecOND_iters .- new_regularization_solve_static_unconstrained_zero_sum_iters
# diff_iters_cesp = SecOND_iters .- cesp_iters

println("Mazumdar: ", size(diff_iters_mazumdar))
println("g_d: ", size(diff_iters_g_d))
println("Simultaneous GDA: ", size(diff_iters_simultaneous_gda))
println("Number of times SecOND failed: ", second_failed)
# println("New Regularization: ", size(diff_iters_new_regularization))
# println("CESP: ", size(diff_iters_cesp))

plot_diff = [diff_iters_mazumdar,  diff_iters_simultaneous_gda, diff_iters_g_d]
# plot_diff = [diff_iters_mazumdar,  diff_iters_simultaneous_gda, diff_iters_g_d, diff_iters_new_regularization, diff_iters_cesp]
# println("Plot Diff: ", size(plot_diff))
# names = ["Mazumdar", "Simultaneous GDA", "g_d", "New Regularization", "CESP"]
# data = [violin(y=plot_diff[y], name=name) for (y, name) in zip([1, 2, 3, 4, 5], names)]
# plot(data..., title="Difference in Value between SecOND and other methods", xlabel="Method", ylabel="Difference in Value", legend=false)

# violin(["\$\\mathrm{LSS}\$" "\$\\mathrm{GDA}\$" "\$g_d\$"], plot_diff, title="Difference in Iterations between \$\\texttt{SecOND}\$ and other methods", xlabel="\$\\mathrm{Method}\$", ylabel="\$\\mathrm{Iterations}_{\\texttt{SecOND}}-\\mathrm{Iterations}_{\\mathrm{Method}}\$", box_visible=true, titlefont=font(10), line=0, alpha = 0.65, legend=false)
# boxplot!(["\$\\mathrm{LSS}\$" "\$\\mathrm{GDA}\$" "\$g_d\$"], plot_diff, title="Difference in Iterations between \$\\texttt{SecOND}\$ and other methods", xlabel="\$\\mathrm{Method}\$", ylabel="\$\\mathrm{Iterations}_{\\texttt{SecOND}}-\\mathrm{Iterations}_{\\mathrm{Method}}\$", box_visible=true, titlefont=font(10), line=(1, :black), fillrange=0, fill = (0.0,), box_width=0.1, legend=false)

violin(["\$\\mathrm{LSS}\$" "\$\\mathrm{GDA}\$" "\$\\mathrm{DND}\$"], plot_diff, xlabel="\$\\mathrm{Baseline\\,\\,Method}\$", ylabel="\$\\mathrm{Difference\\,\\,of\\,\\,Iterations\\,\\,with\\,\\,Baseline}\$", box_visible=true, titlefont=font(10), line=0, alpha = 0.65, legend=false)
boxplot!(["\$\\mathrm{LSS}\$" "\$\\mathrm{GDA}\$" "\$\\mathrm{DND}\$"], plot_diff, xlabel="\$\\mathrm{Baseline\\,\\,Method}\$", ylabel="\$\\mathrm{Difference\\,\\,of\\,\\,Iterations\\,\\,with\\,\\,Baseline}\$", box_visible=true, titlefont=font(10), line=(1, :black), fillrange=0, fill = (0.0,), box_width=0.1, legend=false)


end

example1()