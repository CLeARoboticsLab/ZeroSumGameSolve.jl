using ZeroSumGameSolve

guess = [-15.0, -2.5]
tol = 10e-8
max_iters = 1000000
α = 0.1
n_x = 1
func = twodimexample
sol, value, iters_taken, path = solve_static_unconstrained_zero_sum(guess, func, n_x, tol, max_iters, α)
println("Solution: ", sol)
println("Value: ", value)
println("Iterations: ", iters_taken)