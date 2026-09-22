# Objectives shared by the tests. They are defined on every process so the
# Distributed tests can ship them to workers by name.
@everywhere begin
    quadratic(x) = sum((x .- [0.2, -0.4]) .^ 2)

    rastrigin(x) = 10 * length(x) + sum(x .^ 2 .- 10 .* cos.(2π .* x))

    function income_moments(x)
        ρ, σ = x
        variance = σ^2 / (1 - ρ^2)
        return [variance, ρ * variance, ρ^2 * variance]
    end

    function rough_holes(x)
        if x[1] < -0.35 || (0.2 < x[1] < 0.45 && x[2] < 0.4)
            throw(ModelEvaluationError("no equilibrium"))
        end
        x[2] < -0.7 && return NaN
        # Stair steps are a stand-in for discretization of model moments.
        return sum(round.((x .- [0.6, 0.3]) ./ 0.02) .^ 2) * 0.0004
    end

    function process_objective(x)
        sleep(0.02)
        return Evaluation(sum(x .^ 2); moments=[myid()])
    end

    broken_remote(x) = throw(ErrorException("remote programming bug"))
end

const BOX2 = [(-2, 2), (-2, 2)]
const METHODS = [NelderMeadLocal(), PatternSearchLocal(), NLoptLocal(:LN_BOBYQA),
                 NLoptLocal(:LN_SBPLX), NLoptLocal(:LN_NELDERMEAD)]

# Parsed journal records of a run directory.
journal(dir) = [JSON.parse(line; dicttype=Dict{String,Any})
                for line in eachline(joinpath(dir, "history.jsonl")) if !isempty(strip(line))]

# Read-only view of a run's state, the analogue of querying the database.
history(dir) = TikTak.Store(dir; readonly=true)
