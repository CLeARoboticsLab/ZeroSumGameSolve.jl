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
        mat_Reg = circle_theorem_regularize(D'*D)
        new_x = x - α*inv(mat_Reg)*D'*F
        error = norm(new_x - x)
        x = new_x
        k += 1
    end
    if k == total_iter
        println("Newton's method did not converge!")
    end
    return x, j1(x), -j1(x), k
end

Point, J_1, J_2, iters = solve_zero_sum(rosenbrock, [0.9; 0.9], 10e-9, 100000, 0.5)
println("P1_Strat: ", Point[1])
println("P2_Strat: ", Point[2])
println("J1: ", J_1)
println("J2: ", J_2)
println("Iterations: ", iters)