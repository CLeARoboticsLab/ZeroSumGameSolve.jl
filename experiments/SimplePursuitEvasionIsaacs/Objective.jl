# "Objective function for the simple pursuit evasion game introduced in Page 16, [1]"
# "[1] Isaacs, Rufus. Differential games: a mathematical theory with applications to warfare and pursuit, control and optimization. Courier Corporation, 1999."

# "Pursuer: x, Evader: y"
function C(x, y)
    return 2*cos(x) - cos(y)
end

function S(x, y)
    return 2*sin(x) - sin(y)
end

function t_m(x, y)
    return (C(x, y) + S(x, y))/(C(x, y)^2 + S(x, y)^2)
end

function d_m_squared(x, y)
    return (-1 + C(x, y)*t_m(x, y))^2 + (-1 + S(x, y)*t_m(x, y))^2
end

export objective_pe
function objective_pe(x, y)
    return d_m_squared(x, y) + 0.8*t_m(x, y)
end