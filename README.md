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

