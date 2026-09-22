"""
    TikTak

Restartable, parallel TikTak global optimization for structural estimation.

Accepts a scalar objective or a model that returns moments, screens scrambled
Sobol points, and runs asynchronous derivative-free local searches on
Distributed workers, threads, or inline. Every model evaluation is journaled
before the search proceeds, so runs can be resumed and reused.
"""
module TikTak

using Distributed
using JSON
using LinearAlgebra
using Random: Xoshiro
using SHA: sha256
using Sobol: SobolSeq, next!
using FileWatching.Pidfile: mkpidlock, PidlockedError

export minimize, TikTakConfig, TikTakResult, LocalResult, has_solution, load_estimates,
       MomentObjective, Evaluation, ModelEvaluationError,
       BoxTransform, to_parameters, to_unit,
       LocalMethod, NelderMeadLocal, PatternSearchLocal, NLoptLocal, CustomLocal,
       AbstractExecutor, InlineExecutor, ThreadedExecutor, DistributedExecutor,
       sobol_points

include("objectives.jl")
include("transforms.jl")
include("storage.jl")
include("local.jl")
include("executors.jl")
include("solver.jl")

end # module
