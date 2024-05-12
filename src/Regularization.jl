using ForwardDiff, LinearAlgebra
using ZeroSumGameSolve

export regularization
function regularization(x, func, n_x, tol=1e-8)
    update_step_grad = zero_sum_gradient(x, func, n_x)
    update_step_hess = zero_sum_hessian(x, func, n_x)
    if norm(update_step_grad) < tol
        return zeros(size(update_step_grad))
    else
        mat = update_step_hess + transpose(update_step_hess)
        return inv(circle_theorem_regularize(mat*mat*mat))*transpose(update_step_hess)*update_step_grad
    end
end