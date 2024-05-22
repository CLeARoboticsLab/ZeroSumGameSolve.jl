
export rectangle_projection
function rectangle_projection(x, xlims, ylims)
    proj_x = zeros(2)
    proj_x[1] = clamp(x[1], xlims[1], xlims[2])
    proj_x[2] = clamp(x[2], ylims[1], ylims[2])
    return proj_x
end

export circle_projection
function circle_projection(x, center, radius)
    if norm(x - center) <= radius
        return x
    else
        return center + radius*(x - center)/norm(x - center)
    end
end