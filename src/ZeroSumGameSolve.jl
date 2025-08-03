module ZeroSumGameSolve

# TODO: isolate all deps here
# TODO: move tests into test/
# TODO: rename all non-module files "file_name.jl" instead of "FileName.jl"
# TODO: function names are "FunctionName" if constructors or structs, and "function_name" otherwise, or "function_name!" if they modify any arguments (usually the first argument)

# Write your package code here.
include("LineSearch.jl")
include("ObjectiveFunctions.jl")
include("Gradient.jl")
include("SinglePlayerTest.jl")
include("TwoPlayer.jl")
include("ExampleFunctions.jl")
include("LimitingODE.jl")
include("Projection.jl")
include("Baseline.jl")
include("Regularization.jl")
include("ConstrainedSecond.jl")
include("SolverQRE.jl")
end
