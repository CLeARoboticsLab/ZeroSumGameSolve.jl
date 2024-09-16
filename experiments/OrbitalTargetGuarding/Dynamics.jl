using StaticArrays
using LinearAlgebra

#Dynamics use the Clohessy-Wiltshire equations
export phi
function phi(ω, δt)
    a = ω*δt
    return [
        4-3*cos(a) 0 0 sin(a)/ω 2*(1-cos(a))/ω 0;
        6*(sin(a)-a) 1 0 2*(cos(a)-1)/ω (4*sin(a)-3*a)/ω 0;
        0 0 cos(a) 0 0 sin(a)/ω;
        3*ω*sin(a) 0 0 cos(a) 2*sin(a) 0;
        6*ω*(cos(a)-1) 0 0 -2*sin(a) 4*cos(a)-3 0;
        0 0 -ω*sin(a) 0 0 cos(a);
    ]
end
# B = SMatrix{6, 3}([0 0 0; 0 0 0; 0 0 0; 1 0 0; 0 1 0; 0 0 1])
# C = SMatrix{3, 6}([1 0 0 0 0 0; 0 1 0 0 0 0; 0 0 1 0 0 0])
export CW_Dynamics
struct CW_Dynamics
    ω::Float64
    δt::Float64
    B
    C
end

export unroll_trajectory
function unroll_trajectory(controls, bandit_dynamics::CW_Dynamics, guard_dynamics::CW_Dynamics, bandit_init_state, guard_init_state)
    dim = size(controls)[1]÷2
    bandit_controls = controls[1:dim]
    guard_controls = controls[dim+1:end]
    horizon = size(bandit_controls)[1]÷3
    bandit_states = [bandit_init_state]
    guard_states = [guard_init_state]
    bandit_coods = [bandit_dynamics.C*bandit_init_state]
    guard_coods = [guard_dynamics.C*guard_init_state]
    for time in 1:horizon
        bandit_state = phi(bandit_dynamics.ω, bandit_dynamics.δt) * bandit_states[time] + phi(bandit_dynamics.ω, bandit_dynamics.δt)*bandit_dynamics.B * bandit_controls[3*(time-1)+1:3*time]
        guard_state = phi(guard_dynamics.ω, guard_dynamics.δt) * guard_states[time] + phi(bandit_dynamics.ω, bandit_dynamics.δt)*guard_dynamics.B * guard_controls[3*(time-1)+1:3*time]
        push!(bandit_states, bandit_state)
        push!(guard_states, guard_state)
        push!(bandit_coods, bandit_dynamics.C*bandit_state)
        push!(guard_coods, guard_dynamics.C*guard_state)
    end
    return bandit_coods, guard_coods
end