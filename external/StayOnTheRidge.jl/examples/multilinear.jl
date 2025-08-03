using Revise
using StayOnTheRidge
using Symbolics
using LinearAlgebra

eval_expr(expr, dict) = Symbolics.substitute(expr, dict)

function generate_example(dim1, dim2)
    n = dim1+dim2
    min_coords = collect(1:dim1-1)
    γ = 1e-5
    ϵ = 1e-3
    domain = Default()
    x = [Symbolics.variable(:x,i) for i in 1:n]
    A = 5 .+ 2 .* randn(dim1, dim2)
    A = round.(A, digits=2)
    # display(A)
    f = x[1:dim1]' * A * x[dim1+1:n]
    f = eval_expr(f,Dict(x[dim1] => 1-sum(x[1:dim1-1]), x[n] => 1-sum(x[dim1+1:n-1])))
    for i = dim1+1:n-1
        f = eval_expr(f, Dict(x[i] => x[i-1]))
    end
    n = n-2
    x = [Symbolics.variable(:x,i) for i in 1:n]
    return Config_FD(eval(build_function(f, x)), n, min_coords, γ, ϵ, domain)
end

config = generate_example(3,3);
elapsed = @elapsed min_max, trajectory, m, k = run_dynamics(config)
