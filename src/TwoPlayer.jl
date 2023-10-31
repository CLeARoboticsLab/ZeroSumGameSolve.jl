using ForwardDiff, LinearAlgebra
using ZeroSumGameSolve

function solve_zero_sum(j1, x, tol, total_iter, α=1.0)
    k = 0
    error = tol+1.0
    while k<total_iter && error>tol
        F = [-2*(1-x[1]) - 400*x[1]*(x[2]-x[1]^2), 200*(x[1]^2 - x[2])]  
        D = zeros(2, 2)
        D[1,1] = 2 - 400*x[2]+1200*x[1]^2
        D[1,2] = -400*x[1]
        D[2,1] = 400*x[1]
        D[2,2] = -200
        hess_p1_reg = circle_theorem_regularize([D[1,1]])
        hess_p2_reg = circle_theorem_regularize([-1.0*D[2,2]])
        D[1,1] = hess_p1_reg[1,1]
        D[2, 2] = -1.0*hess_p2_reg[1,1]
        new_x = x - α*inv(D)*F
        error = norm(new_x - x)
        x = new_x
        k += 1
    end
    if k == total_iter
        println("Newton's method did not converge!")
    end
    return x, j1(x), -j1(x), k
end

Point, J_1, J_2, iters = solve_zero_sum(rosenbrock, [20, -30], 10e-5, 100000, 0.5)
println("P1_Strat: ", Point[1])
println("P2_Strat: ", Point[2])
println("J1: ", J_1)
println("J2: ", J_2)
println("Iterations: ", iters)