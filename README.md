# ZeroSumGameSolve

This repo contains code for our submission titled Second-Order Algorithms for Finding Local Nash Equilibria in Zero-Sum Games.

We use Julia for our implementation.
To run this code,
1) Open the Julia REPL.
2) In the REPL, type "]" to open package mode. Then enter the command "activate ." .
3) Exit package mode by hitting backspace.
4) Run "import Pkg, Pkg.instantiate()" to download all dependencies. You may need to run "Pkg.resolve()" before.
5) You are now ready to run experiments.
6) To generate unconstrained toy example results: run "include("experiments\\GANExample\\src\\RandomToyExample.jl")".
7) To generate constrained toy example results: run "include("experiments\\GANExample\\src\\TestConstrained1.jl")"
    and include("experiments\\GANExample\\src\\TestConstrained2.jl").
8) For the GAN example:

* Go to the `GANExample` directory in the terminal: `cd experiments/GANExample/` and start Julia REPL with `julia`.

* Activate the environment via `]` and `activate .`.

* Precompile the package: `using GANExample`.

* To run training for GDA: `GANExample.train_gan_standard()`.

* To run training for LSS: `GANExample.train_gan_zero_sum(; approach = "mazumdar")`.

* To run training for our approach: `GANExample.train_gan_zero_sum(; approach = "ours_optimizer")`.

* To reproduce the result plots: `GANExample.plot_gan_example_comparison()`.

