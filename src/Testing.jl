using ZeroSumGameSolve

guess = [1.0, 2.0]
tol = 10e-5
max_iters = 100000
α = 1.0
n_x = 1
func = twodimexample
sol, value, iters_taken, path = solve_static_unconstrained_zero_sum(guess, func, n_x, tol, max_iters, α)
println("Solution: ", sol)
println("Value: ", value)
println("Iterations: ", iters_taken)