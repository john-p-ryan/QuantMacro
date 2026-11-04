# Criterion profiles: fix one parameter on a grid and reoptimize the others.
#
# Each grid point is a separate, journaled run of `minimize` with the profiled
# parameter fixed. Points are visited outwards from the estimate so that each search
# is warm-started from its neighbour's solution as well as from the estimate.

"""
    ProfileResult

Criterion profile of one parameter, built by [`profile`](@ref). For each grid value
`values[i]` (ascending), `losses[i]` is the smallest loss found with parameter `index`
fixed there and `x[i]` the full parameter vector attaining it (`nothing` without a
feasible point). `delta = losses .- loss` is the increase over `loss`, the loss at
`estimate`. `status` gives the status of each search and `results` its
[`TikTakResult`](@ref). `issues` lists searches that did not finish, whose losses are
only upper bounds on the profile, and grid points whose loss is more than `tolerance`
below the estimate's, which means the estimate is not the global minimum.
"""
struct ProfileResult
    name::String
    index::Int
    values::Vector{Float64}
    losses::Vector{Float64}
    delta::Vector{Float64}
    x::Vector{Union{Nothing,Vector{Float64}}}
    status::Vector{Symbol}
    results::Vector{TikTakResult}
    estimate::Vector{Float64}
    loss::Float64
    tolerance::Float64
    run_dir::String
    issues::Vector{String}
end

_finished(status::Symbol) = status in (:completed, :target_reached)

function _loss_at(objective, x)
    output = objective(copy(x))
    output isa Evaluation && return output.value
    output isa Real && return Float64(output)
    throw(ArgumentError("objective must return a real number or an Evaluation"))
end

function _parameter_index(parameter, names, n)
    if parameter isa Integer && !(parameter isa Bool)
        1 <= parameter <= n || throw(ArgumentError("parameter index $parameter is out of range 1:$n"))
        return Int(parameter)
    end
    names === nothing && throw(ArgumentError("select the parameter by index, or pass names to select it by name"))
    labels = _labels(names, n, "θ", "names")
    index = findfirst(==(string(parameter)), labels)
    index === nothing && throw(ArgumentError("no parameter is named $(parameter)"))
    return index
end

# Saved grid losses are reused on resume, so a model change would silently corrupt
# `delta`; the loss at an unchanged estimate detects it. The record is written once
# the profile has finished, so a failed call leaves the previous one in place.
function _check_estimate_loss(path, estimate, loss)
    isfile(path) || return nothing
    saved = JSON.parse(read(path, String); dicttype=Dict{String,Any})
    if Vector{Float64}(saved["estimate"]) == estimate && !isapprox(Float64(saved["loss"]), loss; rtol=1e-10)
        throw(ArgumentError("the loss at the estimate is $(loss), but the saved profile in $(dirname(path)) " *
                            "has $(saved["loss"]); the model, data, or draws changed, so use a new " *
                            "problem_id and run_dir"))
    end
    return nothing
end

function _record_estimate_loss(path, estimate, loss)
    mkpath(dirname(path))
    write(path, JSON.json(Dict("estimate" => estimate, "loss" => loss)))
    return nothing
end

function _with(x, index, value)
    point = copy(x)
    point[index] = value
    return point
end

"""
    profile(objective, bounds, estimate, parameter, values; kwargs...) -> ProfileResult

Criterion profile `Q̃(v) = min_{θ : θⱼ = v} Q(θ)` of parameter `j = parameter` (an index,
or a name in `names`) on the grid `values`, where `Q` is the loss of `objective` and
`estimate` is the full parameter vector of an estimate within `bounds`. Each grid point is
a separate [`minimize`](@ref) run over the other parameters, with parameter `j` fixed by
equal bounds. Grid points are visited outwards from `estimate[j]`, and each search is
warm-started from `estimate` and from the solution at the neighbouring grid point closer
to the estimate, both with `θⱼ = v`.

A profile is a global diagnostic along one parameter: a flat profile shows weak
identification, an asymmetric one a poor normal approximation, and a loss below the
estimate's shows that the estimation search missed the global minimum. With efficient
weights, [`profile_interval`](@ref) inverts it into a confidence interval.

- `config`: [`TikTakConfig`](@ref) of each search, by default
  `TikTakConfig(n_samples=32, n_local=3)` because the warm starts carry most of the
  information. `max_evals` applies to each grid point. Any `target_value` is dropped,
  since each search must reach the conditional minimum rather than a target.
- `warm_start`: further physical parameter vectors joining the starts of every new grid
  point, with parameter `j` set to the grid value.
- `run_dir`, `problem_id`, `resume`: as in [`minimize`](@ref). Grid point `v` is saved in
  `run_dir/p<j>_<v>` with problem id `"<problem_id>/profile/p<j>=<v>"`. With
  `resume=true`, saved grid points resume and new ones start, so the grid can be refined
  later. A resume fails if the bounds, the algorithm settings, or the targets, weights,
  and scales of a [`MomentObjective`](@ref) differ from the saved runs, and if the loss
  at an unchanged estimate differs from the one saved in `run_dir/estimate-p<j>.json`.
  Other changes to the model, data, or draws are not detected: change `problem_id`.
- `tolerance`: how far a profile loss may fall below the estimate's loss before it is
  reported, by default `1e-6 * max(1, |loss|)`. Local solvers stop at `x_tol` and `f_tol`,
  so the profile at `estimate[j]` itself can be slightly lower.
- Other keywords, such as `screen_objective`, `scale`, `location`, `tail`, `workers`,
  `executor`, and `store_evaluations`, are passed to every call of [`minimize`](@ref).
  Grid points run in sequence; each search uses the workers.

The loss at `estimate` is evaluated once with `objective`, outside any journal.
"""
function profile(objective, bounds, estimate::AbstractVector{<:Real}, parameter, values::AbstractVector{<:Real};
                 names=nothing, config::TikTakConfig=TikTakConfig(n_samples=32, n_local=3), run_dir=nothing,
                 problem_id=nothing, resume::Bool=false, tolerance=nothing, warm_start=nothing, kwargs...)
    lower, upper = _parse_bounds(bounds)
    θ = Vector{Float64}(estimate)
    n = length(lower)
    length(θ) == n || throw(ArgumentError("estimate must have one entry per bound"))
    (all(isfinite, θ) && all(lower .<= θ .<= upper)) ||
        throw(ArgumentError("estimate must be finite and within bounds"))
    j = _parameter_index(parameter, names, n)
    name = _labels(names, n, "θ", "names")[j]
    lower[j] < upper[j] || throw(ArgumentError("parameter $name is fixed; only free parameters can be profiled"))
    grid = sort(unique(Float64.(values) .+ 0.0))   # adding 0.0 maps -0.0 to 0.0
    isempty(grid) && throw(ArgumentError("values must not be empty"))
    (all(isfinite, grid) && all(lower[j] .<= grid .<= upper[j])) ||
        throw(ArgumentError("profile values must be finite and within the bounds of $name"))
    if run_dir === nothing
        resume && throw(ArgumentError("resume requires run_dir"))
        directory = mktempdir(; prefix="tiktak-profile-", cleanup=false)
        problem_id = something(problem_id, "temporary-run")
    else
        directory = abspath(String(run_dir))
        (problem_id isa AbstractString && !isempty(strip(problem_id))) ||
            throw(ArgumentError("provide a nonempty versioned problem_id with run_dir"))
    end

    extra_starts = warm_start === nothing ? Vector{Float64}[] : _warm_rows(warm_start, BoxTransform(bounds))
    point_config = config.target_value === nothing ? config : TikTakConfig(config; target_value=nothing)

    loss = _loss_at(objective, θ)
    isfinite(loss) || throw(ArgumentError("the loss at the estimate is not finite"))
    tol = tolerance === nothing ? 1e-6 * max(1.0, abs(loss)) : Float64(tolerance)
    (isfinite(tol) && tol >= 0) || throw(ArgumentError("tolerance must be nonnegative and finite"))
    record = joinpath(directory, "estimate-p$(j).json")
    _check_estimate_loss(record, θ, loss)

    results = Vector{TikTakResult}(undef, length(grid))
    first_above = searchsortedfirst(grid, θ[j])
    # Walk outwards from the estimate on each side, warm-starting from the neighbour.
    for side in (first_above-1:-1:1, first_above:length(grid))
        previous = θ
        for i in side
            v = grid[i]
            point_bounds = [(k == j ? (v, v) : (lower[k], upper[k])) for k in 1:n]
            point_dir = joinpath(directory, "p$(j)_$(v)")
            point_resume = resume && _initialized(point_dir)
            starts = point_resume ? nothing :
                unique([[_with(θ, j, v), _with(previous, j, v)]; [_with(x, j, v) for x in extra_starts]])
            results[i] = minimize(objective, point_bounds; config=point_config, run_dir=point_dir,
                                  problem_id="$problem_id/profile/p$(j)=$(v)", resume=point_resume,
                                  warm_start=starts, kwargs...)
            has_solution(results[i]) && (previous = results[i].x)
        end
    end

    _record_estimate_loss(record, θ, loss)
    losses = [r.fun for r in results]
    delta = losses .- loss
    status = [r.status for r in results]
    issues = String[]
    for i in eachindex(grid)
        _finished(status[i]) || push!(issues, "the search at $name = $(_short(grid[i])) ended with status " *
                                              ":$(status[i]); its loss only bounds the profile from above")
        delta[i] < -tol && push!(issues, "the loss at $name = $(_short(grid[i])) is $(_short(-delta[i])) below " *
                                         "the estimate's; the estimate is not the global minimum")
    end
    x = Union{Nothing,Vector{Float64}}[r.x === nothing ? nothing : copy(r.x) for r in results]
    return ProfileResult(name, j, grid, losses, delta, x, status, results, θ, loss, tol, directory, issues)
end

"""
    profile_interval(p::ProfileResult; critical=3.841458820694124) -> NamedTuple

Confidence interval `{v : Q̃(v) - Q(θ̂) <= critical}` from a criterion profile, the
inversion of the distance-metric test of `θⱼ = v` (Newey & West 1987). The default
`critical` is the 95% quantile of χ² with one degree of freedom.

The test statistic is asymptotically χ²₁ only when
- the loss is the efficient criterion on the scale of the J statistic,
  `(m - target)'Ω⁻¹(m - target)` with `Ω` the moment-error covariance including
  simulation noise: the objective of [`two_step`](@ref), or weights from
  [`inverse_covariance_weights`](@ref) without regularization;
- the model is correctly specified, so the J test does not reject; and
- neither `θⱼ` nor the reoptimized parameters are near a bound.
Otherwise the interval has no stated coverage. Unlike Wald intervals, it is invariant to
reparametrization and can be asymmetric.

The loss of a search that did not finish is an upper bound on the profile, so such a
grid point counts as inside the set when its increase is at most `critical` and is
skipped otherwise. Crossings between finished points are interpolated linearly in `√ΔQ`,
which is exact for a quadratic profile. Returns `(lower, upper, lower_bounded,
upper_bounded, disjoint)`: an endpoint that is not bounded is the last grid value reached
without crossing `critical`, so extend the grid. `disjoint` is true when grid points
beyond a crossing fall below `critical` again, so the confidence set is not an interval
and the result covers only the part containing the estimate. Throws if the profile
finds a loss below the estimate's (beyond `p.tolerance`); reestimate first.
"""
function profile_interval(p::ProfileResult; critical::Real=3.841458820694124)
    (isfinite(critical) && critical > 0) || throw(ArgumentError("critical must be positive and finite"))
    any(<(-p.tolerance), p.delta) && throw(ArgumentError(
        "the profile finds a lower loss than the estimate; reestimate from the profile minimum first"))
    kept = [i for i in eachindex(p.values) if isfinite(p.delta[i])]
    values = [p.estimate[p.index]; p.values[kept]]
    roots = sqrt.(max.([0.0; p.delta[kept]], 0.0))
    exact = [true; _finished.(p.status[kept])]
    order = sortperm(values)
    values, roots, exact = values[order], roots[order], exact[order]
    centre = findfirst(==(1), order)
    threshold = sqrt(critical)

    # The crossing of √critical walking away from the estimate, or the last value reached.
    function endpoint(indices)
        last = centre
        for i in indices
            # An unfinished search above the threshold says nothing about the profile.
            !exact[i] && roots[i] > threshold && continue
            if roots[i] > threshold
                t = (threshold - roots[last]) / (roots[i] - roots[last])
                return values[last] + t * (values[i] - values[last]), true, i
            end
            last = i
        end
        return values[last], false, nothing
    end
    lower, lower_bounded, below = endpoint(centre-1:-1:1)
    upper, upper_bounded, above = endpoint(centre+1:length(values))
    disjoint = (below !== nothing && any(<=(threshold), roots[1:below-1])) ||
               (above !== nothing && any(<=(threshold), roots[above+1:end]))
    return (; lower, upper, lower_bounded, upper_bounded, disjoint)
end

function Base.show(io::IO, ::MIME"text/plain", p::ProfileResult)
    print(io, "ProfileResult: ", p.name, " on ", length(p.values), " grid points; estimate ",
          _short(p.estimate[p.index]), " with loss ", _short(p.loss))
    rows = [[p.name, "loss", "increase", "status"]]
    for i in eachindex(p.values)
        push!(rows, [_short(p.values[i]), _short(p.losses[i]), _short(p.delta[i]), string(p.status[i])])
    end
    _print_rows(io, rows; left=(4,))
    isempty(p.issues) || print(io, "\n  issues:\n    ", join(p.issues, "\n    "))
    print(io, "\n  run directory: ", p.run_dir)
end

Base.show(io::IO, p::ProfileResult) =
    print(io, "ProfileResult(", p.name, ", ", length(p.values), " grid points)")
