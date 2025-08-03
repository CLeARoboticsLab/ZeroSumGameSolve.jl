using ZeroSumGameSolve, LinearAlgebra, StaticArrays, ReverseDiff, Zygote
include("pegame_utils.jl")

delta = 1.0
dt = 1.0
v_amax=0.8
T = 5
tol = 1e-5
max_iters = Int(1e4)
α = 0.1

pegame = makePEGame(dt, delta, v_amax, T, tol, max_iters, α)

guess = zeros(pegame.prob_dim)
guess[1] = 0.5
guess[2:pegame.n_x] .= 1/(pegame.n_x - 1)
guess[pegame.n_x+1:end] .= pegame.v_amax-0.1
println("Initial guess: ", guess)
func = x -> PEObjective(x, pegame.delta, pegame.dt, pegame.T, pegame.v_amax, pegame.i_iter_start, pegame.num_possible_x0s, pegame.n_x)
println("function", func)
sol, func_val, iters, path = solver_PE(guess, func, pegame.n_x, pegame.tol, pegame.max_iters, pegame.α, pegame.v_amax)
println("Solution: ", sol)
println("Function Value: ", func_val)
println("Iterations: ", iters)
