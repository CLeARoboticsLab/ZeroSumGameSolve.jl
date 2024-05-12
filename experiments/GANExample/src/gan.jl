#============================== Common infrastructure ===============================#


struct GAN
    z_dim::Integer
    generator::Any
    discriminator::Any
end

function construct_training_setup()
    function decoder_gt(z)
        tanh.(1.5z)
    end

    rng = Random.MersenneTwister(1)

    training_config = (;
        optimizer = Optimisers.Adam(0.0001, (0.9, 0.999), 1.0e-8),
        n_epochs = 50000,
        batchsize = 128,
        n_datapoints = 10_000,
        device = cpu,
        time_difference_k = 3, # difference of the update frequency between the generator and the discriminator
    )

    dims = (; dim_x = 1, dim_hidden = 8, dim_z = 1) # dim_x: data dimension dim_z: 
    # construct dataset
    # dataset = randn(rng, dims.dim_z, training_config.n_datapoints) |> decoder_gt |> training_config.device
    sample_distribution = MixtureModel(Normal, [(-3, 1), (3, 1)])
    dataset = rand(rng, sample_distribution, dims.dim_z, training_config.n_datapoints) |> training_config.device
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