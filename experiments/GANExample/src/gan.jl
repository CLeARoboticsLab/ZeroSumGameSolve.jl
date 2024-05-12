
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