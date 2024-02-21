using ZeroSumGameSolve
Point, J_1, iters = solve_zero_sum([0.1, 0.1], 10e-5, 100000, 0.1)
println("P1_Strat: ", Point[1])
println("P2_Strat: ", Point[2])
println("Objective Value ", J_1)
println("Iterations: ", iters)