using ZeroSumGameSolve
using LinearAlgebra

export alpha
function alpha(point, func, n_x)
    hess = zero_sum_hessian(point, func, n_x)
    hess2 = hess'
    mat = hess + hess2
    eigenvalues = eigvals(mat)
    return minimum([1.98*abs(minimum(eigenvalues)), 0.00001])
end