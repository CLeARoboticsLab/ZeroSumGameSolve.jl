# This file contains the definition of the objective function to be minimized.
# Objective function is the toy example from Mazumdar's paper
export twodimexample
function twodimexample(x, y)
    return exp(-0.01*(x^2 + y^2))*((0.3*x^2 + y)^2 + (0.5*y^2 + x)^2)
end

export entropy_regularized_bilinear
function entropy_regularized_bilinear(x, y)
    return x*y + x*log(x) - y*log(y)
end