using ZeroSumGameSolve
using Plots
using Random
using StatsPlots
using LinearAlgebra

function test_constrained2()
    tol = 10e-5
    ball_tol = 1e-2
    max_iters = 50000
    α = 0.009
    n_x = 1
    func = twodimexample
    xlims = [-20, 0]
    ylims = [-13, 5]
    xs = range(xlims...; length = 100)
    ys = range(ylims...; length = 100)
    # circle_center = [0., 0.]
    # circle_radius = 30.0
    false_circle_center = [-10.5, -5.0]
    false_circle_radius = 100.0
    test_circle_center = [-10.5, -5.0]
    test_circle_radius = 5.0
    circle_center = test_circle_center
    circle_radius = test_circle_radius
    θ = range(0, stop=2π, length=1000)
    x_c = test_circle_center[1] .+ test_circle_radius * cos.(θ)
    y_c = test_circle_center[2] .+ test_circle_radius * sin.(θ)
    # plt = contourf(xs, ys, twodimexample; color = :viridis)
    plt = contourf(xs, ys, twodimexample; color=:turbo)
    sol, value, iters_taken, path = constrained_g_d([-11., -3.2], circle_center, circle_radius, func, n_x, tol, max_iters, α)
    sol2, value2, iters_taken2, path2 = constrained_g_d([-11., -3.2], false_circle_center, false_circle_radius, func, n_x, tol, max_iters, α)
    plot!(x_c, y_c, seriestype=:shape, color=:red, label="Constraint Set \$\\mathcal{G}\$", line=(2, 1.0, :grey), alpha = 0.4)
    # plot!(x_c, y_c, seriestype=:line, color=:black, line=(1), alpha = 1.0)
    plot!([x[1] for x in path], [x[2] for x in path], color = :black, show=true, label="\$\\texttt{SeCoND}\$")
    plot!([x[1] for x in path2], [x[2] for x in path2], color = :red, linestyle=:dash, show=true, label="\$g_{d}\$ (ours) without constraints")
    bdr_pt = [-14.94837799712655, -7.282965876810337]
    vector = zero_sum_gradient(bdr_pt, func, n_x)
    vector = [vector[1][1], vector[2][1]]
    vector = 0.4*vector / norm(vector)
    quiver!([bdr_pt[1]], [bdr_pt[2]], quiver=([vector[1]], [vector[2]]), 
       arrow=:closed, c=:blue, label="vector")
    # scatter!([sol[1]], [sol[2]], color=:green, label="Local Nash Equilibrium", markershape=:star5, markersize=10)
    # scatter!([-11.], [-3.2], color=:yellow, label="Initial Point \$\\hat{\\mathbf{z}}\$", markershape=:circle, markersize=5)
    scatter!([bdr_pt[1]], [bdr_pt[2]], color=:yellow, label="Boundary point \$\\mathbf{z}\$", markershape=:circle, markersize=2)
    annotate_pt = bdr_pt + vector
    annotate!([annotate_pt[1]], [annotate_pt[2]-.1], "Direction of \$\\omega(\\mathbf{z})\$")
    xlims!(-15.5, -14.4)
    ylims!(-8.5, -6.5)
    annotate!((sol[1], sol[2]), text("Local Nash Equilibrium", :black))
    println("Solution: ", sol)
end

test_constrained2()
