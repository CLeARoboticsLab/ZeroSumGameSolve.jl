using ZeroSumGameSolve
using LinearAlgebra
using ReverseDiff
using CUDA
using Zygote: Zygote



function norm_w(x, func)
    grads = Zygote.gradient(x) do x
        func(x)
    end
    return 0.5*norm(grads[1])^2
end

export alpha
function alpha(point, func, step, w, J, c=1e-4)
    m = transpose(w)*J*step
    α = 1.0
    update = zeros(size(step))
    for i in 1:size(step)[1]
        update[i] = step[i][1]
    end
    point_new = point - α*update
    while (norm_w(point, func) - norm_w(point_new, func) < c*α*m) && α >= 1e-8
        α = 0.5*α
        point_new = point - α*update
    end

    return α
end