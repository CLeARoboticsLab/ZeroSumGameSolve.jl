export evolve_agent
function evolve_agent(velocity, heading, starting, times, dt)
    trajectory = zeros(2, length(times))
    trajectory[:, 1] = starting
    for i in 2:length(times)
        trajectory[1, i] = trajectory[1, i-1] + velocity*cos(heading)*dt
        trajectory[2, i] = trajectory[2, i-1] + velocity*sin(heading)*dt
    end
    return trajectory
end