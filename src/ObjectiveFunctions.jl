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

export grimmer
function grimmer(x, y)
    return x^4 - 10*(x^2) + 10*x*y -y^4 + 10*(y^2)
end

export norm_grad_grimmer
function norm_grad_grimmer(x, y)
    w1 = 4*x^3 - 20*x + 10*y
    w2 = 4*y^3 - 20*y-10*x
    return sqrt(w1^2 + w2^2)
end