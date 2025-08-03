using ZeroSumGameSolve

struct PEGame
    n_x::Int
    dt::Float64 
    delta::Float64
    v_amax::Float64
    i_iter_start::Int
    prob_dim:: Int
    T::Int
    num_possible_x0s::Int
    tol::Float64
    α::Float64
    max_iters::Int

end

function makePEGame(dt::Float64, delta::Float64, v_amax::Float64, T::Int, tol::Float64, max_iters::Int, α::Float64)
    i_iter_start = Int(round(delta / dt))
    num_possible_x0s = T - 2*i_iter_start
    n_x = 1 + num_possible_x0s
    prob_dim = n_x + T
    return PEGame(n_x, dt, delta, v_amax, i_iter_start, prob_dim, T, num_possible_x0s, tol, α, max_iters)
end



struct PEGamefixed
    n_x::Int
    dt::Float64 
    delta::Float64
    v_amax::Float64
    prob_dim:: Int
    T::Int
    x0::Int
    tol::Float64
    α::Float64
    max_iters::Int
end

function makePEGamefixed(dt::Float64, delta::Float64, v_amax::Float64, T::Int, tol::Float64, max_iters::Int, α::Float64)
    n_x = 1
    prob_dim = n_x + T
    x0 = T*dt-delta
    return PEGamefixed(n_x, dt, delta, v_amax, prob_dim, T, x0, tol, α, max_iters)
end


# export PEgradient
# function PEgradient(point, func)
#     a = SVector{size(point)...}(point)
#     return ReverseDiff.gradient(x->func(x), a)
# end


# ## Gives out
# ## J = [∇_xx,   ∇_xy]
# ##     [∇_yx,   ∇_yy]
# export PEhessian
# function PEhessian(point, func)
#     a = SVector{size(point)...}(point)
#     return ReverseDiff.hessian(x->func(x), a)
# end


# ## Takes the form
# ## w = [∇_x; -∇_y]
# export PEzero_sum_gradient
# function PEzero_sum_gradient(point, func, n_x)
#     grad = PEgradient(point, func)
#     update_step_grad = [grad[1:n_x], -1.0*grad[n_x+1:end]]
#     update_step_grad = SVector{size(update_step_grad)...}(update_step_grad)
#     return update_step_grad
# end


# ## Fundamental matrix of the paper.
# ## J = [∇_xx,   ∇_xy]
# ##     [-∇_yx, -∇_yy]
# export PEzero_sum_true_hessian
# function PEzero_sum_true_hessian(point, func, n_x)
#     hess = PEhessian(point, func)
#     ∇_xx = hess[1:n_x, 1:n_x]
#     ∇_xy = hess[1:n_x, n_x+1:end]
#     ∇_yx = -1.0*hess[n_x+1:end, 1:n_x]
#     ∇_yy = -1.0*hess[n_x+1:end, n_x+1:end]
#     update_step_hess = [∇_xx ∇_xy; ∇_yx ∇_yy]
#     update_step_hess = SMatrix{size(update_step_hess)...}(update_step_hess)
# end
## ∇f(x)
export PEgradient
function PEgradient(point::AbstractVector, func::Function)
    grad = Zygote.gradient(func, point)[1]
    return SVector{length(grad)}(grad)
end

## ∇²f(x)
export PEhessian
function PEhessian(point::AbstractVector, func::Function)
    hess = Zygote.hessian(func, point)
    return SMatrix{length(point), length(point)}(hess)
end

## w = [∇_x; -∇_y]
export PEzero_sum_gradient
function PEzero_sum_gradient(point::AbstractVector, func::Function, n_x::Int)
    grad = PEgradient(point, func)
    update_step_grad = vcat(grad[1:n_x], -grad[n_x+1:end])
    return SVector{length(update_step_grad)}(update_step_grad)
end

## J = [∇_xx,   ∇_xy;
##      -∇_yx, -∇_yy]
export PEzero_sum_true_hessian
function PEzero_sum_true_hessian(point::AbstractVector, func::Function, n_x::Int)
    hess = PEhessian(point, func)
    ∇_xx = hess[1:n_x, 1:n_x]
    ∇_xy = hess[1:n_x, n_x+1:end]
    ∇_yx = -hess[n_x+1:end, 1:n_x]
    ∇_yy = -hess[n_x+1:end, n_x+1:end]
    update_step_hess = [∇_xx ∇_xy; ∇_yx ∇_yy]
    return SMatrix{size(update_step_hess)...}(update_step_hess)
end


export solver_PE
function solver_PE(guess, func, n_x, tol, max_iters, α, v_amax)
    x = guess
    prob_dim = length(guess)
    k = 0
    error = tol+1.0
    path = [x]
    while k<max_iters && error>tol
        # println("k = ", k)
        update_step_grad = PEzero_sum_gradient(x, func, n_x)
        println("update_step_grad: ", update_step_grad)
        update_step_hess = PEzero_sum_true_hessian(x, func, n_x)
        static_array_update = PE_regularization_g_d(update_step_grad, update_step_hess, n_x, prob_dim)
        # update = zeros(size(static_array_update)) for i in 1:size(static_array_update)[1]
        #     update[i] = static_array_update[i][1]
        # end
        update = Vector(static_array_update)
        w = Vector(update_step_grad)
        m = (transpose(update)*w/norm(w))*w
        m = m / norm(w)
        x_new = enforce_PE_constraints(x-α*m, n_x, v_amax)
        push!(path, x_new)
        error = norm(x_new - x)
        # println(x)
        x = x_new
        k += 1
    end
    println("iters: ", k)
    if k==max_iters
        println("constrained g_d did not converge!")
        println("Error: ", error)
    end
    return x, func(x), k, path 
end

function project_to_simplex(x::AbstractVector{<:Real})
    n = length(x)
    u = sort(x, rev=true)
    cssv = cumsum(u) .- 1
    rho = findlast(i -> u[i] > cssv[i] / i, 1:n)
    theta = cssv[rho] / rho
    return max.(x .- theta, 0)
end

function enforce_PE_constraints(x, n_x, v_amax)
    #clip x[1] to be in [0, 1]
    x[1] = clamp(x[1], 0.0, 1.0)
    # project x[2:n_x] to simplex
    x[2:n_x] = project_to_simplex(x[2:n_x])
    x[n_x+1:end] = clamp.(x[n_x+1:end], -v_amax, v_amax)
    return x
end

export PEObjective
function PEObjective(x, delta, dt, T, v_amax, i_iter_start, num_possible_x0s, n_x)
    return sum(x[i-i_iter_start+1]*(x[1]*(norm(sum(x[n_x+1:n_x+ i - i_iter_start])*dt - i*dt + delta)^2 + norm(sum(sqrt.(v_amax^2 .- x[n_x+1:n_x + i - i_iter_start].^2))*dt)^2 ) + (1-x[1])*(norm(sum(x[n_x+1:n_x+ i + i_iter_start])*dt - i*dt - delta)^2 + norm(sum(sqrt.(v_amax^2 .- x[n_x+1:n_x + i + i_iter_start].^2))*dt)^2)) for i in 1+i_iter_start:T-i_iter_start)
    # total = zero(eltype(x))
    # for i in (1+i_iter_start):(T - i_iter_start)
    #     xi = x[i - i_iter_start + 1]
    #     p = x[1]
        
    #     # Left side
    #     left_vels = x[n_x+1 : n_x + i - i_iter_start]
    #     left_pos = dt * sum(left_vels)
    #     left_term1 = norm(left_pos - i*dt + delta)^2
    #     left_term2 = norm(sum(sqrt.(v_amax^2 .- left_vels.^2)) * dt)^2
    #     left_cost = left_term1 + left_term2

    #     # Right side
    #     right_vels = x[n_x+1 : n_x + i + i_iter_start]
    #     right_pos = dt * sum(right_vels)
    #     right_term1 = norm(right_pos - i*dt - delta)^2
    #     right_term2 = norm(sum(sqrt.(v_amax^2 .- right_vels.^2)) * dt)^2
    #     right_cost = right_term1 + right_term2

    #     # Weighted average cost
    #     total += xi * (p * left_cost + (1 - p) * right_cost)
    # end
    # return total
    # vel = x[n_x+1:end]
    # weight = x[1]
    # total = 0.0

    # for i in (1 + i_iter_start):(T - i_iter_start)
    #     idx = i - i_iter_start
    #     xi = x[idx + 1]
        
    #     # Velocity ranges
    #     v_range1 = vel[1:idx]
    #     v_range2 = vel[idx+1:2idx]

    #     # Sum terms
    #     sum_v1 = sum(v_range1) * dt
    #     sum_v2 = sum(v_range2) * dt

    #     sqrt_term1 = sqrt.(v_amax^2 .- v_range1.^2) .* dt
    #     sqrt_term2 = sqrt.(v_amax^2 .- v_range2.^2) .* dt

    #     # Avoid norm^2: use sum of squares
    #     loss1 = sum((sum_v1 - i*dt + delta).^2) + sum(sqrt_term1.^2)
    #     loss2 = sum((sum_v2 - i*dt - delta).^2) + sum(sqrt_term2.^2)

    #     total += xi * (weight * loss1 + (1 - weight) * loss2)
    # end

    # return total
end


function PE_regularization_g_d(update_step_grad, update_step_hess, n_x, problem_dim)
    beta = zeros(problem_dim, problem_dim)
    if isposdef(update_step_hess[1:n_x, 1:n_x])
        beta[1:n_x, 1:n_x] = LinearAlgebra.I(n_x)
    end
    if isposdef(update_step_hess[n_x+1:end, n_x+1:end])
        beta[n_x+1:end, n_x+1:end] = -LinearAlgebra.I(problem_dim - n_x)
    end
    return inv(circle_theorem_regularize(transpose(update_step_hess)*update_step_hess*(update_step_hess+transpose(update_step_hess)))+ beta) * transpose(update_step_hess)* update_step_grad
end

#x is now [p, vel_1, ..., vel_T]
export PEObjectivefixed
function PEObjectivefixed(x, delta, dt, T, v_amax, x0, n_x, radius)
    t1 = Int((x0-delta)/dt)
    t2 = T
    return x[1]*(norm(x0-delta + sum(x[2:end-2*Int(delta/dt)]) - dt*t1)^2 + norm(sum(sqrt.(v_amax^2 .- x[2:end-2*Int(delta/dt)].^2))*dt)^2 + sum(norm(x0-delta +sum(x[2:1+i])*dt - i*dt)^2 + norm(sum(sqrt.(v_amax^2 .- x[2:1+i].^2))*dt)^2 - radius^2 for i in 1:T)) + (1-x[1])*(x0+delta + norm(sum(x[2:end]) - dt*t2)^2 + norm(sum(sqrt.(v_amax^2 .- x[2:end].^2))*dt)^2 + sum(norm(x0 + delta + sum(x[2:1+i])*dt - i*dt)^2 + norm(sum(sqrt.(v_amax^2 .- x[2:1+i].^2))*dt)^2 - radius^2 for i in 1:T))
end

export solver_PE_fixed
function solver_PE_fixed(guess, func, n_x, tol, max_iters, α, v_amax)
    x = guess
    prob_dim = length(guess)
    k = 0
    error = tol+1.0
    path = [x]
    while k<max_iters && error>tol
        # println("k = ", k)
        update_step_grad = PEzero_sum_gradient(x, func, n_x)
        # println("update_step_grad: ", update_step_grad)
        update_step_hess = PEzero_sum_true_hessian(x, func, n_x)
        static_array_update = PE_regularization_g_d(update_step_grad, update_step_hess, n_x, prob_dim)
        # update = zeros(size(static_array_update)) for i in 1:size(static_array_update)[1]
        #     update[i] = static_array_update[i][1]
        # end
        update = Vector(static_array_update)
        w = Vector(update_step_grad)
        if norm(w) > tol
            if check_if_fixed_active(x, n_x, v_amax, tol)
                m = (transpose(update)*w/norm(w))*w
                m = m / norm(w)
                x_new = enforce_fixed_PE_constraints(x-α*m, n_x, v_amax)
            else
                x_new = enforce_fixed_PE_constraints(x-α*update, n_x, v_amax)
            end
            push!(path, x_new)
            error = norm(x_new - x)
            # println("x_new: ", x_new)
            x = x_new
        else
            println("Gradient norm is below tolerance, stopping.")
            break
        end
        k += 1
    end
    println("iters: ", k)
    if k==max_iters
        println("constrained g_d did not converge!")
        println("Error: ", error)
    end
    return x, func(x), k, path 
end

function enforce_fixed_PE_constraints(x, n_x, v_amax)
    #clip x[1] to be in [0, 1]
    x[1] = clamp(x[1], 0.0, 1.0)
    # project x[2:n_x] to simplex
    # x[2:n_x] = project_to_simplex(x[2:n_x])
    x[n_x+1:end] = clamp.(x[n_x+1:end], -v_amax, v_amax)
    return x
end

function check_if_fixed_active(x, n_x, v_amax, tol_active)
    # Check if the first element is in [0, 1]
    result = false
    if x[1] < 0.0 + tol_active || x[1] > 1.0 - tol_active
        result = true
    end
    # Check if the velocities are within bounds
    result = result || any(abs.(x[n_x+1:end]) .> v_amax - tol_active)
    return result
end


function get_traj(x_vels, v_max)
    T = length(x_vels)
    traj = zeros(T, 2)
    for i in 1:T
        traj[i, 1] = sum(x_vels[1:i])
        traj[i, 2] = sum(sqrt.(v_max^2 .- x_vels[1:i].^2))
    end
    return traj
end

export plot_traj

function plot_traj(x_vels, v_max, x0, delta, p)
    traj = get_traj(x_vels, v_max)

    # Base trajectory plot
    plt = plot(traj[:, 1], traj[:, 2], 
        xlabel="Position_x", 
        ylabel="Velocity", 
        title="Trajectory", 
        label="Trajectory", 
        linewidth=3)

    # Barograph: vertical bars at x0 - delta and x0 + delta
    bar_xs = [x0 - delta, x0 + delta]
    bar_ys = [p, 1 - p]
    bar!(bar_xs, bar_ys, label="Barograph", bar_width=1e-3, legend=:topright)
    
    return plt
end
