#================================= Train GAN using our zero sum solver ============================# 


function train_gan_zero_sum(; set_up = construct_training_setup(), training_log_sample_size = 1000, approach = "ours")
    # generator = JLD2.load("data/generator.jld2")["generator"]
    # discriminator = JLD2.load("data/discriminator.jld2")["discriminator"]
    gan = setup_gan(set_up)
    generator = gan.generator
    discriminator = gan.discriminator
    fixed_ϵ = rand(set_up.rng, Distributions.Normal(), gan.z_dim, training_log_sample_size) # fixed noise samples for tracking training loss
    fixed_data = set_up.dataset[:, 1:training_log_sample_size] # fixed data samples for tracking training loss
    params_generator, reconstruct_generator = destructure(generator)
    params_discriminator, reconstruct_discriminator = destructure(discriminator)
    generator_optimizer_setup = Optimisers.setup(set_up.training_config.optimizer, params_generator) # generator optimizer
    discriminator_optimizer_setup = Optimisers.setup(set_up.training_config.optimizer, params_discriminator) # discriminator optimizer
    losses = Vector{Float64}()
    for epoch in 1:set_up.training_config.n_epochs
        println("Epoch $epoch")
        ii = 0
        for mini_batch in set_up.data_batch_iterator
            num_samples = size(mini_batch)[2]
            ϵ = rand(set_up.rng, Distributions.Normal(), gan.z_dim, num_samples) # noise samples
            loss = get_objective_function_for_zero_sum_solve(generator, discriminator; ϵ, mini_batch) # loss function over the **mini_batch**
            params_generator, reconstruct_generator = destructure(generator)
            params_discriminator, reconstruct_discriminator = destructure(discriminator)
            params_gan = vcat(params_generator, params_discriminator)
            dim_params_generator = size(params_generator)[1]
            # newton direction computation
            println(epoch, " ", ii)
            if approach == "ours"
                zero_sum_sol = ZeroSumGameSolve.new_reg_GAN(params_gan, loss, dim_params_generator, 1e-7, 1, epoch)
            elseif approach == "mazumdar"
                zero_sum_sol = ZeroSumGameSolve.GAN_mazumdar_two_timescale_approximation(params_gan, loss, dim_params_generator, 1e-7, 1, epoch)
            end
            # direction_generator = deepcopy(zero_sum_sol[1][1:dim_params_generator] - params_generator)
            # direction_discriminator = deepcopy(zero_sum_sol[1][(dim_params_generator + 1):end] - params_discriminator)

            # update GAN parameters
            # generator_optimizer_setup, params_generator = Optimisers.update(generator_optimizer_setup, params_generator, direction_generator)
            # discriminator_optimizer_setup, params_discriminator = Optimisers.update(discriminator_optimizer_setup, params_discriminator, direction_discriminator)
            generator = reconstruct_generator(zero_sum_sol[1][1:dim_params_generator])
            discriminator = reconstruct_discriminator(zero_sum_sol[1][(dim_params_generator + 1):end])
            ii += 1
        end
        current_loss = (sum(log.(discriminator(fixed_data) .+ 1e-6)) 
        + sum(log.(1 .- discriminator(generator(fixed_ϵ)) .+ 1e-6))) / training_log_sample_size
        @info "loss: $(current_loss)"
        push!(losses, current_loss)
        if epoch % 1000 == 0
            plot_loss_curve(losses; approach, epoch)
            plot_generated_samples(generator; set_up, gan.z_dim, approach, epoch)
            jldsave("data/"*approach*string(epoch)*"_generator"*(now() |> string)*".jld2"; generator)
            jldsave("data/"*approach*string(epoch)*"_discriminator"*(now() |> string)*".jld2"; discriminator)
            jldsave("data/"*approach*string(epoch)*"_losses"*(now() |> string)*".jld2"; losses)
        end
    end
    plot_loss_curve(losses; approach)
    plot_generated_samples(generator; set_up, gan.z_dim, approach)
    jldsave("data/"*approach*"_generator"*(now() |> string)*".jld2"; generator)
    jldsave("data/"*approach*"_discriminator"*(now() |> string)*".jld2"; discriminator)
    jldsave("data/"*approach*"_losses"*(now() |> string)*".jld2"; losses)
end

function get_objective_function_for_zero_sum_solve(generator, discriminator; ϵ, mini_batch)
    params_generator, reconstruct_generator = destructure(generator)
    params_discriminator, reconstruct_discriminator = destructure(discriminator)
    dim_params_generator = size(params_generator)[1]
    parameters = vcat(params_generator, params_discriminator)
    function loss(parameters)
        generator = reconstruct_generator(parameters[1:dim_params_generator])
        discriminator = reconstruct_discriminator(parameters[(dim_params_generator + 1):end])
        fake_data = generator(ϵ)
        (sum(log.(discriminator(mini_batch) .+ 1e-6)) + sum(log.(1 .- discriminator(fake_data) .+ 1e-6))) / size(mini_batch)[2]
    end
end

#================= Train GAN using standard way ==============================#

function train_gan_standard(; set_up = construct_training_setup(), training_log_sample_size = 1000)
    gan = setup_gan(set_up)
    generator = gan.generator
    discriminator = gan.discriminator
    fixed_ϵ = rand(set_up.rng, Distributions.Normal(), gan.z_dim, training_log_sample_size) # fixed noise samples for tracking training loss
    fixed_data = set_up.dataset[:, 1:training_log_sample_size] # fixed data samples for tracking training loss
    generator_optimizer_setup = Optimisers.setup(set_up.training_config.optimizer, generator) # generator optimizer
    discriminator_optimizer_setup = Optimisers.setup(set_up.training_config.optimizer, discriminator) # discriminator optimizer
    losses = Vector{Float64}()
    for epoch in 1:set_up.training_config.n_epochs
        println("Epoch $epoch")
        ii = 0
        for mini_batch in set_up.data_batch_iterator
            num_samples = size(mini_batch)[2]
            if ii % (set_up.training_config.time_difference_k + 1) != 0
                # update discriminator
                ϵ = rand(set_up.rng, Distributions.Normal(), gan.z_dim, num_samples) # noise samples
                fake_data = generator(ϵ)
                # explicit style of gradient computation
                grads = Zygote.gradient(discriminator) do model
                    loss = get_discriminator_loss(model)
                    loss(mini_batch, fake_data)
                end
                ∇m = grads[1]
                discriminator_optimizer_setup, discriminator = Optimisers.update(discriminator_optimizer_setup, discriminator, ∇m)
            else
                # update generator
                ϵ = rand(set_up.rng, Distributions.Normal(), gan.z_dim, num_samples) # noise samples
                # explicit style of gradient computation
                grads = Zygote.gradient(generator) do model
                    loss = get_generator_loss(model, discriminator)
                    loss(ϵ)
                end
                ∇m = grads[1]
                generator_optimizer_setup, generator = Optimisers.update(generator_optimizer_setup, generator, ∇m)
            end
            ii += 1
        end
        current_loss = (sum(log.(discriminator(fixed_data) .+ 1e-6)) 
        + sum(log.(1 .- discriminator(generator(fixed_ϵ)) .+ 1e-6))) / training_log_sample_size
        @info "loss: $(current_loss)"
        push!(losses, current_loss)
    end
    plot_loss_curve(losses)
    plot_generated_samples(generator; set_up, gan.z_dim)
    jldsave("data/gda_generator"*(now() |> string)*".jld2"; generator)
    jldsave("data/gda_discriminator"*(now() |> string)*".jld2"; discriminator)
    jldsave("data/gda_losses"*(now() |> string)*".jld2"; losses)
end

function get_generator_loss(generator, discriminator)
    function loss(ϵ)
        generated_fake_data = generator(ϵ)
        # sum(log.(1 .- discriminator(generated_fake_data) .+ 1e-6)) / size(ϵ)[2] # According to Goodfellow et al., use log(G(ϵ)) instead for improved gradient signal
        - sum(log.(discriminator(generated_fake_data) .+ 1e-6)) / size(ϵ)[2]
    end
end

function get_discriminator_loss(discriminator)
    function loss(real_data, fake_data)
        - (sum(log.(discriminator(real_data) .+ 1e-6)) + sum(log.(1 .- discriminator(fake_data) .+ 1e-6))) / size(real_data)[2]
    end
end

#============================== Common infrastructure ===============================#


struct GAN
    z_dim::Integer
    generator::Any
    discriminator::Any
end

function construct_training_setup()
    rng = Random.MersenneTwister(1)

    training_config = (;
        optimizer = Optimisers.Adam(0.0001, (0.9, 0.999), 1.0e-8),
        n_epochs = 10000,
        batchsize = 128,
        n_datapoints = 10_000,
        device = cpu,
        time_difference_k = 3, # difference of the update frequency between the generator and the discriminator
    )

    dims = (; dim_x = 2, dim_hidden = 2, dim_z = 2) # dim_x: data dimension dim_z: 
    # construct dataset
    # dataset = randn(rng, dims.dim_z, training_config.n_datapoints) |> decoder_gt |> training_config.device
    angle_interval = 2*π / 8
    radius = 1
    sample_distributions = map(1:8) do ii
        θ = (ii - 1) * angle_interval
        MvNormal([radius * cos(θ), radius * sin(θ)], (1e-4) .* I(2))
    end
    sample_distribution = MixtureModel(MvNormal[sample_distributions[ii] for ii in 1:length(sample_distributions)])
    dataset = rand(rng, sample_distribution, training_config.n_datapoints) |> training_config.device
    data_batch_iterator = Flux.Data.DataLoader(dataset; training_config.batchsize, shuffle = true, rng)

    (; rng, training_config, dims, dataset, data_batch_iterator)
end

function setup_gan(set_up = construct_training_setup(); generator = nothing, discriminator = nothing)
    discriminator = isnothing(discriminator) ? Chain(
        Dense(set_up.dims.dim_x, set_up.dims.dim_hidden, relu; init = glorot_uniform(set_up.rng)),
        Dense(set_up.dims.dim_hidden, set_up.dims.dim_hidden, relu; init = glorot_uniform(set_up.rng)),
        Dense(set_up.dims.dim_hidden, 1, sigmoid; init = glorot_uniform(set_up.rng)),
    ) : discriminator
    generator = isnothing(generator) ? Chain(
        Dense(set_up.dims.dim_z, set_up.dims.dim_hidden, relu; init = glorot_uniform(set_up.rng)),
        Dense(set_up.dims.dim_hidden, set_up.dims.dim_hidden, relu; init = glorot_uniform(set_up.rng)),
        Dense(set_up.dims.dim_hidden, set_up.dims.dim_x; init = glorot_uniform(set_up.rng)),
    ) : generator
    GAN(set_up.dims.dim_z, generator, discriminator) |> set_up.training_config.device
end