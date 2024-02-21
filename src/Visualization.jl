using ZeroSumGameSolve
using Plots
using Random

function create_anim(
    f,
    path,
    xlims,
    ylims,
    file_name = joinpath(pwd(), randstring(12) * ".gif");
    xbounds = xlims,
    ybounds = ylims,
    fps = 15,
)
    xs = range(xlims...; length = 100)
    ys = range(ylims...; length = 100)
    plt = contourf(xs, ys, f; color = :viridis)

    # add constraints if provided
    if !(xbounds == xlims && ybounds == ylims)
        x_rect = [xbounds[1]; xbounds[2]; xbounds[2]; xbounds[1]; xbounds[1]]
        y_rect = [ybounds[1]; ybounds[1]; ybounds[2]; ybounds[2]; ybounds[1]]

        plot!(x_rect, y_rect; line = (2, :dash, :red), label="")
    end

    # add an empty plot
    plot!(Float64[], Float64[]; line = (4, :arrow, :black), label = "")

    # extract the last plot series
    plt_path = plt.series_list[end]

    # create the animation and save it
    anim = Animation()
    for i in 1:size(path)[1]
        push!(plt_path, path[i][1], path[i][2]) # add a new point
        frame(anim)
    end
    gif(anim, file_name; fps = fps, show_msg = false)
    return nothing
end


guess = [10.0, 5.0]
# guess = [0.0, 0.0]
tol = 10e-7
max_iters = 10000000
α = 0.00001
n_x = 1
func = twodimexample
sol, value, iters_taken, path = solve_static_unconstrained_zero_sum(guess, func, n_x, tol, max_iters, α)
println("Solution: ", sol)
println("Value: ", value)
println("Iterations: ", iters_taken)
create_anim(func, path, [-20, 20], [-20, 20], "ex1fast.gif")