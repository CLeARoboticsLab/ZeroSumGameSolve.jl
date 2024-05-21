using ZeroSumGameSolve
using Plots
using Random
using StatsPlots


function test_constrained()
    tol = 10e-5
    ball_tol = 1e-2
    max_iters = 50000
    α = 0.009
    n_x = 1
    func = twodimexample
    xlims = [-20, 20]
    ylims = [-20, 20]
    xs = range(xlims...; length = 100)
    ys = range(ylims...; length = 100)
    circle_center = [0., 0.]
    circle_radius = 30.0
    test_circle_center = [-15, -5.0]
    test_circle_radius = 5.0
    θ = range(0, stop=2π, length=1000)
    x_c = test_circle_center[1] .+ test_circle_radius * cos.(θ)
    y_c = test_circle_center[2] .+ test_circle_radius * sin.(θ)
    plt = contourf(xs, ys, twodimexample; color = :viridis)
    sol, value, iters_taken, path = constrained_g_d([-10.2, -3.2], circle_center, circle_radius, func, n_x, tol, max_iters, α)
    plot!(x_c, y_c, seriestype=:shape, color=:red, label="Constraint", line=(1, :black), alpha = 0.1)
    plot!([x[1] for x in path], [x[2] for x in path], color = :black, show=true)
    println("Solution: ", sol)
end

test_constrained()
