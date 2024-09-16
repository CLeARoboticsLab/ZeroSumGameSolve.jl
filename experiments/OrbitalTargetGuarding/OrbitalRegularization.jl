export circle_theorem_regularize
function circle_theorem_regularize(x)
    n = size(x)[1]
    ϵ = zeros(n, n)
    for i in 1:n
        c = x[i, i]
        r = norm(x[i, 1:n], 1) - abs(c)
        if c >= 0
            if c - r < 0
                ϵ[i, i] = r - c
            end
        else
            ϵ[i, i] = r + abs(c)
        end
    end
    return x + ϵ + 5.0*I(n)
end

export orbital_regularization_dnd
function orbital_regularization_dnd(update_step_grad, update_step_hess)
    n = size(update_step_grad)[1]÷2
    beta = zeros(2*n, 2*n)
    if isposdef(update_step_hess[1:n, 1:n])
        for i in 1:n
            beta[i, i] = 1.0
        end
    end
    if isposdef(-1.0*update_step_hess[n+1:end, n+1:end])
        for i in 1:n
            beta[i+n, i+n] = -1.0
        end
    end
    return inv(circle_theorem_regularize(transpose(update_step_hess)*update_step_hess*(update_step_hess+transpose(update_step_hess)))+ beta) * transpose(update_step_hess)* update_step_grad
end

export orbital_regularization_SecOND
function orbital_regularization_SecOND(update_step_grad, update_step_hess, ball_tol)
    n = size(update_step_grad)[1]÷2
    if norm(update_step_grad) < ball_tol
        beta = zeros(n, n)
        if isposdef(update_step_hess[1:n, 1:n])
            for i in 1:n
                beta[i, i] = 1.0
            end
        end
        if isposdef(-1.0*update_step_hess[n+1:end, n+1:end])
            for i in 1:n
                beta[i+n, i+n] = -1.0
            end
        end
        return inv(circle_theorem_regularize(transpose(update_step_hess)*update_step_hess*(update_step_hess+transpose(update_step_hess))) + beta) * transpose(update_step_hess)* update_step_grad, false
    else
        return inv(transpose(update_step_hess)*update_step_hess + 0.1*LinearAlgebra.I(size(update_step_grad)[1]))*transpose(update_step_hess)*update_step_grad, true
    end
    
end

