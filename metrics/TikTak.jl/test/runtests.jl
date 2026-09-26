using Test
using Distributed
using LinearAlgebra
using JSON
using TikTak
using NLopt

# Two worker processes for the Distributed tests. Workers do not inherit the
# active project, so it is passed explicitly; the same applies in real runs.
addprocs(2; exeflags="--project=$(Base.active_project())")
@everywhere using TikTak, NLopt

include("helpers.jl")

@testset verbose = true "TikTak" begin
    include("test_transform.jl")
    include("test_objective.jl")
    include("test_config.jl")
    include("test_storage.jl")
    include("test_local.jl")
    include("test_minimize.jl")
    include("test_resume.jl")
    include("test_staged.jl")
    include("test_parallel.jl")
end

rmprocs(workers())
