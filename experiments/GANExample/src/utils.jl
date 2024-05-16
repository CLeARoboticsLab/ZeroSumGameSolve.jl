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