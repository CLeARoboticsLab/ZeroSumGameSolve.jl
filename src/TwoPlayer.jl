using ForwardDiff, LinearAlgebra
using ZeroSumGameSolve, Symbolics

function solve_zero_sum(j1, xₒ, tol, total_iter, α=1.0)
    k = 0
    error = tol+1.0

    @variables x[1:length(xₒ)]

    y = [x...]

    f = j1(x)
    fx = eval(build_function(f,x,expression = Val{false}))

    F = Symbolics.gradient(f,y)
    F[2] = -F[2]
    Fx = eval(build_function(F,x,expression = Val{false})[1])
    

    D = Symbolics.jacobian(F,x)
    Dx = eval(build_function(D,x,expression = Val{false})[1])
    
    while k<total_iter && error>tol
        # F = [∂j1/∂x1, ∂j1/∂x2] through ForwardDiff, then change sign for P2

        currF = Fx(xₒ)
        
        # Calculate D i.e,e jacobian of F

        currD = Dx(xₒ)
           
        hess_p1_reg = circle_theorem_regularize([currD[1,1]])
        hess_p2_reg = circle_theorem_regularize([-1.0*currD[2,2]])
        currD[1,1] = hess_p1_reg[1,1]
        currD[2, 2] = -1.0*hess_p2_reg[1,1]
        new_x = xₒ - α*inv(currD)*currF
        error = norm(new_x - xₒ)
        xₒ = new_x
        k += 1
    end
    if k == total_iter
        println("Newton's method did not converge!")
    end
    return xₒ, fx(xₒ), -fx(xₒ), k
end


print("Time to Converge: ")
Point, J_1, J_2, iters = @time solve_zero_sum(PaperExample, [0,0], 10e-10, 100000, 0.5)

println("P1_Strat: ", Point[1])
println("P2_Strat: ", Point[2])
println("J1: ", J_1)
println("J2: ", J_2)
println("Iterations: ", iters)

# @time solve_zero_sum(rosenbrock,[20, -30], 10e-5, 100000, 0.5)
