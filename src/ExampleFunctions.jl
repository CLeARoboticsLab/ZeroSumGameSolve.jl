using ForwardDiff, LinearAlgebra
using ZeroSumGameSolve, Symbolics

export objective_function
function objective_function(x, y)
    return (exp(-0.01*(x^2 + y^2)))*((0.3*x^2 + y)^2 + (0.5*y^2 + x)^2)
end

export x_gradient_function
function x_gradient_function(x, y)
    ∇_x_f = -0.02*x*objective_function(x, y) + exp(-0.01*(x^2 + y^2))*((1.2*x*(0.3*x^2 + y)) + (2*x*(0.5*y^2 + x)))
    return ∇_x_f
end

export y_gradient_function
function y_gradient_function(x, y)
    ∇_y_f = -0.02*y*objective_function(x, y) + exp(-0.01*(x^2 + y^2))*((2*(0.3*x^2 + y)) + (2*y*(0.5*y^2 + x)))
    return ∇_y_f
end

export hessian_function
function hessian_function(x, y)
    H_f = zeros(2, 2)
    H_f[1, 1] = -0.02*(objective_function(x, y) + x*x_gradient_function(x, y)) -0.02*x*exp(-0.01*(x^2 + y^2))*(1.2*x*(0.3*x^2 + y) + 2(0.5*y^2 + x)) + exp(-0.01*(x^2 + y^2))*(1.08*x^2 + 1.2*y + 2)
    H_f[1, 2] = -0.02*y*x_gradient_function(x,y) -0.02*x*exp(-0.01*(x^2 + y^2))*(2*(0.3*x^2 + y) + 2*y*(0.5*y^2 + x)) + exp(-0.01*(x^2 + y^2))*(5.2*x + 2*y)
    H_f[2, 1] = -H_f[1, 2]
    H_f[2, 2] = -0.02*(objective_function(x, y) + y*y_gradient_function(x, y)) -0.02*y*exp(-0.01*(x^2 + y^2))*(2*(0.3*x^2 + y) + 2*y*(0.5*y^2 + x)) + exp(-0.01*(x^2 + y^2))*(2+3*y^2+2*x)
    H_f[2, 2] = -H_f[2, 2]
    return H_f
end

export solve_zero_sum
function solve_zero_sum(guess, tol, total_iter, α=1.0)
    k = 0
    error = tol+1.0
    x = guess[1]
    y = guess[2]
    z = [x, y]
    while k<total_iter && error>tol
        # F = [∂j1/∂x1, ∂j1/∂x2] through ForwardDiff, then change sign for P2
        F = [x_gradient_function(x, y), -1.0*y_gradient_function(x, y)]
        # Calculate D i.e,e jacobian of F
        H = hessian_function(x, y)
        H_reg_11 = circle_theorem_regularize([H[1,1]])
        H_reg_12 = circle_theorem_regularize([H[2,2]])
        H_reg = [H_reg_11[1] H[1,2]; H[2,1] H_reg_12[1]]
        new_z = z - α*inv(H_reg)*F
        error = norm(new_z - z)
        z = new_z
        k += 1
    end
    if k == total_iter
        println("Newton's method did not converge!")
    end
    val = objective_function(z[1], z[2])
    return z, val, k
end