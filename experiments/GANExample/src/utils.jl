function plot_loss_curve(losses; approach = "gda", epoch = nothing)
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1], title = "Training Loss Curve", xlabel = "Epoch", ylabel = "Loss")
    Makie.lines!(ax, Vector{Float64}(1:length(losses)), losses)
    if !isnothing(epoch)
        Makie.save("data/"*approach*string(epoch)*"_training_loss"*(now() |> string)*".png", fig)
    else
        Makie.save("data/"*approach*"_training_loss"*(now() |> string)*".png", fig)
    end
end

function plot_generated_samples(generator; set_up, z_dim, approach = "gda", epoch = nothing)
    ϵ = rand(set_up.rng, Distributions.Normal(), z_dim, 10000) 
    generated_samples = generator(ϵ)
    fig = Makie.Figure(resolution = (1600, 800), fontsize = 35)
    colors = [colorant"rgba(105, 105, 105, 0.65)", colorant"rgba(254, 38, 37, 0.65)"]
    ax = Makie.Axis(fig[1, 1], title="ground truth vs. learned distribution", 
        xlabel = "data value", ylabel = "probability density", 
        spinewidth=3, xlabelsize = 40, ylabelsize = 40)
    Makie.density!(ax, set_up.dataset |> vec, color = colors[1], strokearound = true, strokewidth = 3, 
        strokecolor = colorant"rgba(105, 105, 105, 1.0)", label = "ground truth")
    Makie.density!(ax, generated_samples |> vec, color = colors[2], strokearound = true, strokewidth = 3, 
        strokecolor = colorant"rgba(254, 38, 37, 1.0)", label = "GAN generated")
    Makie.axislegend(ax)
    if !isnothing(epoch)
        Makie.save("data/"*approach*string(epoch)*"_generated_samples"*(now() |> string)*".png", fig)
    else
        Makie.save("data/"*approach*"_generated_samples"*(now() |> string)*".png", fig)
    end
end

function load_file_from_partial_name(partial_name; directory = "data/")
    files = readdir(directory)
    matching_files = filter(f -> occursin(partial_name, f), files)
    if length(matching_files) == 0
        error("No file found with the partial name: $partial_name")
    elseif length(matching_files) > 1
        error("Multiple files found with the partial name: $partial_name. Please specify further.")
    else
        full_filename = joinpath(directory, matching_files[1])
    end

    data = JLD2.load(full_filename)
end

function plot_gan_example_comparison(; 
    directory = "data/",
    epoch_interval = 3000,
    img_per_row = 5,
    solver_names = ["gda", "mazumdar", "ours_optimizer"],
    mapped_solver_names = ["GDA", "LSS", "SecOND"],
    set_up = construct_training_setup(),
)
    ϵ = rand(set_up.rng, Distributions.Normal(), set_up.dims.dim_z, 10000) 
    epochs = map(2:img_per_row) do ii
        ii * epoch_interval
    end

    colors = [colorant"rgba(105, 105, 105, 0.65)", colorant"rgba(255, 145, 46, 0.65)", colorant"rgba(62, 173, 217, 0.65)", colorant"rgba(254, 38, 37, 0.55)"]
    colors_frame = [colorant"rgba(105, 105, 105, 1.0)", colorant"rgba(255, 145, 46, 1.0)", colorant"rgba(62, 173, 217, 1.0)", colorant"rgba(254, 38, 37, 1.0)"]
    legend_elems = [[Makie.MarkerElement(color = colors[ii], marker=:circle, markersize = 60, strokecolor = colors_frame[ii])] for ii in 1:length(colors)]

    
    fig = Makie.Figure(; size = (img_per_row * 475, length(solver_names) * 500 + 150), fontsize = 35)
    for ii in 1:length(solver_names)
        approach = solver_names[ii]
        mapped_approach = mapped_solver_names[ii]
        for jj in 1:length(epochs)
            epoch = epochs[jj]
            generator = load_file_from_partial_name(approach * string(epoch) * "_generator")["generator"]
            discriminator = load_file_from_partial_name(approach * string(epoch) * "_discriminator")["discriminator"]
            generated_samples = generator(ϵ)

            ax = Makie.Axis(fig[ii, jj], title= "$mapped_approach : " * string(epoch) * " iters.", 
                xlabel = "data value", ylabel = "prob. density", 
                spinewidth=3, xlabelsize = 40, ylabelsize = 40)
            ax.titlesize = 45
            Makie.density!(ax, set_up.dataset |> vec, color = colors[1], strokearound = true, strokewidth = 3, 
                strokecolor = colors_frame[1], label = "ground truth")
            Makie.density!(ax, generated_samples |> vec, color = colors[ii + 1], strokearound = true, strokewidth = 3, 
                strokecolor = colors_frame[ii + 1], label = "GAN generated")
            Makie.ylims!(ax, 0, 0.5)
            # Makie.axislegend(ax) 
        end
    end
    Makie.rowgap!(fig.layout, 1, Relative(0.1))
    Makie.rowgap!(fig.layout, 2, Relative(0.1))
    # Makie.save(directory * "gan_comparison_"*string(epoch_interval)*"_"*string(img_per_row)*".png", fig)
    Makie.Legend(fig[0, 1:img_per_row-1], legend_elems, ["  GT (ground truth) ", "  GDA (baseline) ", "  LSS (baseline) ", "  SecOND (ours) "], framevisible = false, 
        orientation = :horizontal, tellwidth = false, tellheight = true)
    Makie.save(directory * "gan_comparison.png", fig)
end

function plot_gan_comparison()
    img_per_row_lst = [7]
    epoch_interval_lst = [3000]
    for img_per_row in img_per_row_lst
        for epoch_interval in epoch_interval_lst
            plot_gan_example_comparison(; img_per_row, epoch_interval)
        end
    end
end