using ForwardDiff, LinearAlgebra
using ZeroSumGameSolve

export regularization
function regularization(x, update_step_grad, update_step_hess, tol=1e-8)
    if norm(update_step_grad) < tol
        return [@SVector [0.0] for i in 1:size(update_step_grad)[1]]
    else
        mat = update_step_hess + transpose(update_step_hess)
        return inv(circle_theorem_regularize(mat*mat*mat))*transpose(update_step_hess)*update_step_grad
    end
end