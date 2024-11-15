using ZeroSumGameSolve
using Plots
using Random
using StatsPlots
using LinearAlgebra

function test_qre()
    tol = 10e-10
    max_iters = 50000
    α = 0.009
    n_x = 1
    func = entropy_regularized_bilinear
    guess = [0.1, 0.9]
    sol, value, iters_taken, sol_path = solver_qre(guess, func, n_x, tol, max_iters, α)
    xlims = [1e-9 5]
    ylims = [1e-9, 5]
    xs = range(xlims...; length = 100)
    ys = range(ylims...; length = 100)
    plt = contourf(xs, ys, func; color=:turbo, aspect_ratio=1.0)
    xs_simplex = [0, 1]
    ys_simplex = 1 .- xs_simplex
    plot!(xs_simplex, ys_simplex, label="Probability Simplex", lw=2, legend=:topright)
    # plot!(x_c, y_c, seriestype=:line, color=:black, line=(1), alpha = 1.0)
    plot!([x[1] for x in sol_path], [x[2] for x in sol_path], color = :black, show=true, label="SeCoND")
    vector = zero_sum_gradient(sol, func, n_x)
    println("omega: ", vector)
    vector = [vector[1][1], vector[2][1]]
    vector = 0.4*vector / norm(vector)
    quiver!([sol[1]], [sol[2]], quiver=([vector[1]], [vector[2]]), 
       arrow=:closed, c=:blue, label="vector")
    scatter!([sol[1]], [sol[2]], color=:green, label="Local Nash Equilibrium", markershape=:star5, markersize=7)
    scatter!([guess[1]], [guess[2]], color=:yellow, label="Initial Point", markershape=:circle, markersize=3)
    # scatter!([-11.], [-3.2], color=:yellow, label="Initial Point \$\\hat{\\mathbf{z}}\$", markershape=:circle, markersize=5)
    xlims!(1e-5, 2)
    ylims!(1e-5, 2)
    annotate!([1.05], [1.0], "Direction of \$\\omega(\\mathbf{z})\$")
    println("Solution: ", sol)
    savefig("data/qre_plots/qre1.png")
end

test_qre()
