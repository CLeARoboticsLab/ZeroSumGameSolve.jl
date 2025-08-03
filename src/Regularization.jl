using ForwardDiff, LinearAlgebra
using ZeroSumGameSolve
using LinearAlgebra
using Arpack

export regularization
function regularization(x, update_step_grad, update_step_hess, tol=1e-8)
    if norm(update_step_grad) < tol
        return [@SVector [0.0] for i in 1:size(update_step_grad)[1]]
    else
        mat = update_step_hess + transpose(update_step_hess)
        return inv(circle_theorem_regularize(mat*mat*mat))*transpose(update_step_hess)*update_step_grad
    end
end

export regularization_SecOND
function regularization_SecOND(update_step_grad, update_step_hess, ball_tol)
    if norm(update_step_grad) < ball_tol
        b1 = 0
        b2 = 0
        if update_step_hess[1, 1] > 0
            b1 = 1.0
        end
        if update_step_hess[2, 2] < 0
            b2 = -1.0
        end
        beta = [b1 0.0; 0.0 b2]
        return inv(circle_theorem_regularize(transpose(update_step_hess)*update_step_hess*(update_step_hess+transpose(update_step_hess))) + beta) * transpose(update_step_hess)* update_step_grad, false
    else
        return inv(transpose(update_step_hess)*update_step_hess + 0.1*LinearAlgebra.I(2))*transpose(update_step_hess)*update_step_grad, true
    end
    
end

export regularization_g_d
function regularization_g_d(update_step_grad, update_step_hess)
    b1 = 0
    b2 = 0
    if update_step_hess[1, 1] > 0
        b1 = 1.0
    end
    if update_step_hess[2, 2] > 0
        b2 = -1.0
    end
    beta = [b1 0.0; 0.0 b2]
    return inv(circle_theorem_regularize(transpose(update_step_hess)*update_step_hess*(update_step_hess+transpose(update_step_hess)))+ beta) * transpose(update_step_hess)* update_step_grad
end

export regularization_g_d_GAN
function regularization_g_d_GAN(update_step_grad, update_step_hess, n_x)
    b1 = 0.0
    b2 = 0.0
    n_y = size(update_step_hess)[1] - n_x
    λ_x = eigs(update_step_hess[1:n_x, 1:n_x], nev=1, which=:SR, ritzvec=false)[1]
    λ_y = eigs(update_step_hess[n_x+1:end, n_x+1:end], nev=1, which=:LR, ritzvec=false)[1]
    if λ_x > 0
        b1 = 1.0
    end
    if λ_y < 0
        b2 = -1.0
    end
    beta = [b1*LinearAlgebra.I(n_x) zeros(n_x, n_y); zeros(n_y, n_x) b2*LinearAlgebra.I(n_y)]
    return inv(circle_theorem_regularize(transpose(update_step_hess)*update_step_hess*(update_step_hess+transpose(update_step_hess)))+ beta) * transpose(update_step_hess)* update_step_grad
end

export regularization_SecOND_GAN
function regularization_SecOND_GAN(update_step_grad, update_step_hess, n_x, ball_tol)
    if norm(update_step_grad) < ball_tol
        b1 = 0.0
        b2 = 0.0
        n_y = size(update_step_hess)[1] - n_x
        λ_x = eigs(update_step_hess[1:n_x, 1:n_x], nev=1, which=:SR, ritzvec=false)[1]
        λ_y = eigs(update_step_hess[n_x+1:end, n_x+1:end], nev=1, which=:LR, ritzvec=false)[1]
        if λ_x > 0
            b1 = 1.0
        end
        if λ_y < 0
            b2 = -1.0
        end
        beta = [b1*LinearAlgebra.I(n_x) zeros(n_x, n_y); zeros(n_y, n_x) b2*LinearAlgebra.I(n_y)]
        return inv(circle_theorem_regularize(transpose(update_step_hess)*update_step_hess*(update_step_hess+transpose(update_step_hess)))+ beta) * transpose(update_step_hess)* update_step_grad
    else
        return inv(transpose(update_step_hess)*update_step_hess + 0.1*LinearAlgebra.I(size(update_step_hess)[1]))*transpose(update_step_hess)*update_step_grad
    end
    
end