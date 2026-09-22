# Small correctness checks, not evidence of performance on large structural models.
#
# Run from the TikTak.jl folder:
#     julia --project=. examples/simple_examples.jl --workers 2 [--output tiktak-runs/simple]
#
# Fits a quadratic, the multimodal Rastrigin function, an analytic AR(1) income
# process with an unbounded parameter, and a staircase objective with
# infeasible regions. Use a different --output directory to run them again.
using Distributed
using JSON

function parse_arguments(args)
    options = Dict{String,Any}("workers" => 0, "output" => "tiktak-runs/simple")
    i = 1
    while i <= length(args)
        if args[i] == "--workers"
            options["workers"] = parse(Int, args[i+1])
            i += 2
        elseif args[i] == "--output"
            options["output"] = args[i+1]
            i += 2
        else
            error("unknown argument $(args[i]); use --workers N and --output DIR")
        end
    end
    return options
end

const OPTIONS = parse_arguments(ARGS)
# Workers do not inherit the active project; pass it so `using TikTak` works there.
OPTIONS["workers"] > 0 && addprocs(OPTIONS["workers"]; exeflags="--project=$(Base.active_project())")

@everywhere using TikTak

# Objectives must be defined on every process that evaluates them.
@everywhere begin
    quadratic(x) = sum((x .- [0.2, -0.4]) .^ 2)

    rastrigin(x) = 10 * length(x) + sum(x .^ 2 .- 10 .* cos.(2π .* x))

    """Stationary AR(1) variance and first two autocovariances."""
    function income_moments(x)
        ρ, σ = x
        (abs(ρ) >= 1 || σ < 0) && throw(ModelEvaluationError("no stationary income distribution"))
        variance = σ^2 / (1 - ρ^2)
        return [variance, ρ * variance, ρ^2 * variance]
    end

    function rough_holes(x)
        if x[1] < -0.35 || (0.2 < x[1] < 0.45 && x[2] < 0.4)
            throw(ModelEvaluationError("no equilibrium"))
        end
        x[2] < -0.7 && return NaN
        return sum(round.((x .- [0.6, 0.3]) ./ 0.02) .^ 2) * 0.0004
    end
end

function main()
    target = income_moments([0.65, 0.2])
    cases = [
        ("quadratic", quadratic, [(-2, 2), (-2, 2)],
         (n_samples=16, n_local=4), (;)),
        ("rastrigin", rastrigin, [(-5.12, 5.12), (-5.12, 5.12)],
         (n_samples=256, n_local=32, local_max_evals=180, seed=7), (;)),
        ("ar1_moments", MomentObjective(income_moments, target; scales=target), [(0, 0.98), (0, Inf)],
         (n_samples=64, n_local=8, local_max_evals=250, x_tol=1e-7), (scale=[1, 0.2],)),
        ("rough_holes", rough_holes, [(-1, 1), (-1, 1)],
         (n_samples=128, n_local=16, local_method=PatternSearchLocal(), local_max_evals=150, x_tol=1e-4), (;)),
    ]
    summary = Dict{String,Any}()
    for (name, objective, bounds, settings, options) in cases
        config = TikTakConfig(; settings...)
        result = minimize(objective, bounds; config=config, run_dir=joinpath(OPTIONS["output"], name),
                          problem_id="example-$name-v1", options...)
        summary[name] = TikTak.to_dict(result)
        println(rpad(name, 14), " loss=", result.fun, "  x=", result.x, "  evaluations=", result.n_evals,
                "  failures=", result.n_failed, "  ", result.status)
    end
    open(joinpath(OPTIONS["output"], "summary.json"), "w") do io
        JSON.json(io, summary; pretty=true)
        write(io, '\n')
    end
end

main()
