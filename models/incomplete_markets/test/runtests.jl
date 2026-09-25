# Tests for the Aiyagari (1994) solvers in models/incomplete_markets. Run from the repository root with
#     julia --project=. models/incomplete_markets/test/runtests.jl

using Test
using LinearAlgebra, SparseArrays
using Spline

# Every solver file defines the same names (Primitives, Results, solve_model, ...), so each gets its own module.
module GridSearch
    using Parameters, LinearAlgebra, Optim, SparseArrays
    include(joinpath(@__DIR__, "..", "Aiyagari_gridsearch.jl"))
end

module EGM
    using Parameters, LinearAlgebra, Optim, SparseArrays, Spline
    include(joinpath(@__DIR__, "..", "Aiyagari_EGM.jl"))
end

module InterpVFI
    using Parameters, LinearAlgebra, Optim, SparseArrays, Spline
    include(joinpath(@__DIR__, "..", "Aiyagari_VFI_interp.jl"))
end

include("helpers.jl")

@testset verbose = true "Aiyagari" begin
    include("test_utility.jl")
    include("test_primitives.jl")
    include("test_household.jl")
    include("test_distribution.jl")
    include("test_equilibrium.jl")
end
