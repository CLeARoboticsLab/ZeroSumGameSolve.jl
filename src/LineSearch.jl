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
function alpha(point, func, step, w, J, c)
    m = transpose(w)*J*step
    α = 1.0
    while norm_w(point, func) - norm_w(point - α*step, func) < c*α*m
        α = 0.5*α
    end

    return α
end