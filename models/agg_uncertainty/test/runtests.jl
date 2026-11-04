# Tests for the aggregate uncertainty solvers in models/agg_uncertainty. Run from the repository root with
#     julia --project=. -t auto models/agg_uncertainty/test/runtests.jl
# (threads only speed up the EGM step of KS_EGM_hist.jl).

using Test
using LinearAlgebra, SparseArrays, Statistics, Random
using Spline

# KS.jl and KS_EGM_hist.jl define the same names (Primitives, Results, ...), so each gets its own module.
# The three linear methods share the Aiyagari steady state and are loaded together, as in methods_comparison.jl.
module KSVFI
    include(joinpath(@__DIR__, "..", "KS.jl"))
end

module KSEGM
    include(joinpath(@__DIR__, "..", "KS_EGM_hist.jl"))
end

module Linear
    include(joinpath(@__DIR__, "..", "BKM_transition.jl"))
    include(joinpath(@__DIR__, "..", "Reiter.jl"))
    include(joinpath(@__DIR__, "..", "SSJ2.jl"))
    include(joinpath(@__DIR__, "..", "SSJ_fake_news.jl"))
end

include("helpers.jl")

@testset verbose = true "Aggregate uncertainty" begin
    include("test_KS_VFI.jl")
    include("test_KS_EGM.jl")
    include("test_BKM.jl")
    include("test_Reiter.jl")
    include("test_SSJ.jl")
    include("test_SSJ_fake_news.jl")
    include("test_methods_agree.jl")
end
