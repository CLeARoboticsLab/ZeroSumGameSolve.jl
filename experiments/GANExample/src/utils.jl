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
    solver_names = ["gda", "ours_optimizer", "mazumdar"],
    set_up = construct_training_setup(),
)
    ϵ = rand(set_up.rng, Distributions.Normal(), set_up.dims.dim_z, 10000) 
    epochs = map(1:img_per_row) do ii
        ii * epoch_interval
    end

    colors = [colorant"rgba(105, 105, 105, 0.65)", colorant"rgba(254, 38, 37, 0.65)"]

    fig = Makie.Figure(; size = (img_per_row * 800, length(solver_names) * 400), fontsize = 35)
    for ii in 1:length(solver_names)
        approach = solver_names[ii]
        for jj in 1:length(epochs)
            epoch = epochs[jj]
            generator = load_file_from_partial_name(approach * string(epoch) * "_generator")["generator"]
            discriminator = load_file_from_partial_name(approach * string(epoch) * "_discriminator")["discriminator"]
            generated_samples = generator(ϵ)

            ax = Makie.Axis(fig[ii, jj], title=approach * " " * string(epoch) * " iterations", 
                xlabel = "data value", ylabel = "probability density", 
                spinewidth=3, xlabelsize = 40, ylabelsize = 40)
            Makie.density!(ax, set_up.dataset |> vec, color = colors[1], strokearound = true, strokewidth = 3, 
                strokecolor = colorant"rgba(105, 105, 105, 1.0)", label = "ground truth")
            Makie.density!(ax, generated_samples |> vec, color = colors[2], strokearound = true, strokewidth = 3, 
                strokecolor = colorant"rgba(254, 38, 37, 1.0)", label = "GAN generated")
            # Makie.axislegend(ax) 
        end
    end
    Makie.save(directory * "gan_comparison.png", fig)
end