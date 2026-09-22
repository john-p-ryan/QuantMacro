# A restartable server entry point; replace `income_moments` with your model.
#
#     julia --project=. examples/estimate_ar1.jl --run-dir /local/disk/ar1-run --workers 16 --max-evals 3000
#     julia --project=. examples/estimate_ar1.jl --run-dir /local/disk/ar1-run --workers 16 --max-evals 6000 --resume
#
# On a cluster, replace `addprocs(n)` with the ClusterManagers.jl manager for
# your scheduler (for example `addprocs(SlurmManager(n))`); the coordinator owns
# the run directory, so workers need no shared filesystem.
using Distributed

function parse_arguments(args)
    options = Dict{String,Any}(
        "run-dir" => nothing, "workers" => parse(Int, get(ENV, "SLURM_CPUS_PER_TASK", "1")) - 1,
        "max-evals" => 3000, "max-seconds" => nothing, "resume" => false)
    i = 1
    while i <= length(args)
        key = lstrip(args[i], '-')
        if key == "resume"
            options["resume"] = true
            i += 1
        elseif key in ("run-dir", "workers", "max-evals", "max-seconds")
            value = args[i+1]
            options[key] = key == "run-dir" ? value : key == "max-seconds" ? parse(Float64, value) : parse(Int, value)
            i += 2
        else
            error("unknown argument $(args[i])")
        end
    end
    options["run-dir"] === nothing && error("--run-dir is required")
    return options
end

const OPTIONS = parse_arguments(ARGS)
OPTIONS["workers"] > 0 && addprocs(OPTIONS["workers"]; exeflags="--project=$(Base.active_project())")

@everywhere using TikTak
@everywhere function income_moments(x)
    ρ, σ = x
    (abs(ρ) >= 1 || σ < 0) && throw(ModelEvaluationError("no stationary income distribution"))
    variance = σ^2 / (1 - ρ^2)
    return [variance, ρ * variance, ρ^2 * variance]
end

function main()
    target = income_moments([0.65, 0.2])
    objective = MomentObjective(income_moments, target; scales=target)
    config = TikTakConfig(n_samples=64, n_local=8, max_evals=OPTIONS["max-evals"],
                          max_seconds=OPTIONS["max-seconds"], local_max_evals=250, x_tol=1e-7)
    result = minimize(objective, [(0, 0.98), (0, Inf)]; scale=[1, 0.2], config=config,
                      run_dir=OPTIONS["run-dir"], problem_id="analytic-ar1-v1", resume=OPTIONS["resume"],
                      callback=r -> println("local searches done: ", r.n_local_completed, "  best: ", r.fun))
    println(result.status, ": loss=", result.fun, ", parameters=", result.x, ", calls=", result.n_evals)
    println("Saved to ", result.run_dir)
end

main()
