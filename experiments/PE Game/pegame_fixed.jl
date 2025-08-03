using ZeroSumGameSolve, LinearAlgebra, StaticArrays, ReverseDiff, Zygote, Plots
include("pegame_utils.jl")

delta = 1.0
dt = 1.0
v_amax=0.8
T = 5
tol = 1e-5
max_iters = Int(1e4)
α = 0.1
radius = 1

pegamefixed = makePEGamefixed(dt, delta, v_amax, T, tol, max_iters, α)

guess = zeros(pegamefixed.prob_dim)
guess[1] = 0.5
println("Initial guess: ", guess)
guess[pegamefixed.n_x+1:end] .= 0.75
println("Initial guess: ", guess)
func = x -> PEObjectivefixed(x, pegamefixed.delta, pegamefixed.dt, pegamefixed.T, pegamefixed.v_amax, pegamefixed.x0, pegamefixed.n_x, radius)
println("function", func)
sol, func_val, iters, path = solver_PE_fixed(guess, func, pegamefixed.n_x, pegamefixed.tol, pegamefixed.max_iters, pegamefixed.α, pegamefixed.v_amax)
println("Solution: ", sol)
println("Function Value: ", func_val)
println("Iterations: ", iters)

plot_traj(sol[2:end], pegamefixed.v_amax, pegamefixed.x0, pegamefixed.delta, sol[1])
