# Descriptive diagnostics of a fit and of the local minima found by a search.

function _labels(names, n, prefix, label)
    names === nothing && return ["$prefix$i" for i in 1:n]
    length(names) == n || throw(ArgumentError("$label must have one entry per element ($n)"))
    return String.(string.(names))
end

_short(x::Real) = isfinite(x) ? string(round(Float64(x); sigdigits=4)) : string(x)

# Aligned table rows below a header line; columns in `left` are left-aligned.
function _print_rows(io::IO, rows::Vector{Vector{String}}; left=(1,))
    widths = [maximum(length(row[j]) for row in rows) for j in eachindex(rows[1])]
    for row in rows
        cells = [j in left ? rpad(cell, widths[j]) : lpad(cell, widths[j]) for (j, cell) in enumerate(row)]
        print(io, "\n  ", rstrip(join(cells, "  ")))
    end
end

"""
    FitSummary

Moment-by-moment decomposition of a minimum-distance loss, built by
[`fit_summary`](@ref). With `g = (moments - target) ./ scales` and weights `W`,
`contributions = g .* (W * g)` sum to `loss = g'Wg`. Under a full weight matrix
each moment receives its own term plus half of every cross term with the other
moments, so a contribution can be negative.
"""
struct FitSummary
    names::Vector{String}
    target::Vector{Float64}
    moments::Vector{Float64}
    errors::Vector{Float64}
    scaled_errors::Vector{Float64}
    contributions::Vector{Float64}
    loss::Float64
end

"""
    fit_summary(objective::MomentObjective, moments; names=nothing) -> FitSummary

Named discrepancies between model `moments` and the objective's targets, and
each moment's contribution to the loss. No model evaluation is performed.
`names` defaults to `"m1"`, `"m2"`, ... The attribution rule is that of
[`FitSummary`](@ref): squared transformed residuals are not attributed to moments.
"""
function fit_summary(objective::MomentObjective, moments::AbstractVector{<:Real}; names=nothing)
    n_moments = length(objective.target)
    length(moments) == n_moments ||
        throw(ArgumentError("moments must have one entry per target moment ($n_moments)"))
    labels = _labels(names, n_moments, "m", "names")
    model_moments = Vector{Float64}(moments)
    errors = model_moments .- objective.target
    scaled_errors = errors ./ objective.scales
    contributions = scaled_errors .* _multiply(objective.weights, scaled_errors)
    return FitSummary(labels, copy(objective.target), model_moments, errors, scaled_errors,
                      contributions, sum(contributions))
end

function Base.show(io::IO, ::MIME"text/plain", s::FitSummary)
    print(io, "FitSummary: loss = ", _short(s.loss), " over ", length(s.names), " moments")
    share = s.loss == 0 ? zeros(length(s.names)) : 100 .* s.contributions ./ s.loss
    rows = [["moment", "target", "model", "error", "scaled", "contribution", "share %"]]
    for i in eachindex(s.names)
        push!(rows, [s.names[i], _short(s.target[i]), _short(s.moments[i]), _short(s.errors[i]),
                     _short(s.scaled_errors[i]), _short(s.contributions[i]), _short(share[i])])
    end
    _print_rows(io, rows)
end

Base.show(io::IO, s::FitSummary) =
    print(io, "FitSummary(loss=", _short(s.loss), ", ", length(s.names), " moments)")

"""
    LocalMinimum

A distinct local minimum found by one or more local searches, in unit-box
coordinates (`unit`) and physical coordinates (`x`). `searches` are the indices
of the local searches that ended here and `n_converged` how many of them
reported convergence. `near_bound` lists physical parameter indices within the
tolerance of a bound. `moments` are present when the objective recorded them,
and `fit` when the run used a [`MomentObjective`](@ref).
"""
struct LocalMinimum
    x::Vector{Float64}
    unit::Vector{Float64}
    fun::Float64
    searches::Vector{Int}
    n_converged::Int
    boundary_distance::Float64
    near_bound::Vector{Int}
    moments::Union{Nothing,Vector{Float64}}
    fit::Union{Nothing,FitSummary}
end

"""
    SearchSummary

Distinct local minima (`minima`, ordered by loss) and search counts from a saved
run; see [`search_summary`](@ref). `n_infeasible` counts completed searches that
never found a finite point, `n_incomplete` searches cut off by the global budget,
and `messages` counts the local solver's termination messages.
"""
struct SearchSummary
    minima::Vector{LocalMinimum}
    parameter_names::Vector{String}
    n_evals::Int
    n_failed::Int
    n_completed::Int
    n_converged::Int
    n_infeasible::Int
    n_incomplete::Int
    messages::Dict{String,Int}
end

# Free-parameter indices of the saved bounds; unit coordinate k belongs to `free[k]`.
function _free_indices(specification)
    bounds = specification["transform"]["bounds"]
    parse_bound(v) = v isa AbstractString ? parse(Float64, v) : Float64(v)
    return [i for (i, pair) in enumerate(bounds) if parse_bound(pair[1]) != parse_bound(pair[2])]
end

function _saved_objective(specification)
    saved = specification["moments"]
    saved === nothing && return nothing
    weights = saved["weights"]
    weights = !isempty(weights) && first(weights) isa AbstractVector ?
        permutedims(reduce(hcat, [Vector{Float64}(row) for row in weights])) : Vector{Float64}(weights)
    return MomentObjective(identity, Vector{Float64}(saved["target"]); weights=weights,
                           scales=Vector{Float64}(saved["scales"]))
end

"""
    search_summary(run_dir; x_tol=1e-3, bound_tol=1e-3,
                   parameter_names=nothing, moment_names=nothing) -> SearchSummary
    search_summary(result::TikTakResult; kwargs...)

Cluster the completed local searches of a saved run into distinct local minima.
Searches are visited in order of loss and join the first minimum whose best
point is within `x_tol` in every unit-box coordinate; otherwise they start a new
one. Losses are not compared, since searches that stop at slightly different
points of one basin differ in loss. Parameters whose unit-box coordinate is
within `bound_tol` of 0 or 1 are reported as near a bound. For an infinite bound
the unit-box edge is the finite tail cutoff, so a minimum flagged there has run
away towards the cutoff.

This is a descriptive, local diagnostic. It helps separate a poor search (few
rediscoveries of the best minimum, unconverged searches, minima at bounds) from
poor fit (see the `fit` of each minimum) before standard errors are interpreted;
it does not certify global optimality. The run directory is read without
modification.
"""
function search_summary(run_dir::AbstractString; x_tol::Real=1e-3, bound_tol::Real=1e-3,
                        parameter_names=nothing, moment_names=nothing)
    isfile(joinpath(run_dir, JOURNAL_NAME)) || throw(ArgumentError("no run history found in $run_dir"))
    store = Store(run_dir; readonly=true)
    specification = get(store, "specification")
    specification === nothing && throw(ArgumentError("run history in $run_dir has no saved specification"))
    free = _free_indices(specification)
    n_parameters = length(specification["transform"]["bounds"])
    labels = _labels(parameter_names, n_parameters, "θ", "parameter_names")
    objective = _saved_objective(specification)

    rows = local_rows(store)
    completed = [row.result for row in rows if row.status === :done && row.result !== nothing]
    feasible = sort!([r for r in completed if r.fun !== nothing && isfinite(r.fun)]; by=r -> (r.fun, r.index))
    messages = Dict{String,Int}()
    for r in completed
        messages[r.message] = get(messages, r.message, 0) + 1
    end

    members = Vector{Vector{LocalResult}}()
    for r in feasible
        home = findfirst(group -> maximum(abs, r.unit .- group[1].unit) <= x_tol, members)
        home === nothing ? push!(members, [r]) : push!(members[home], r)
    end

    minima = map(members) do group
        rep = group[1]
        distance = minimum(min.(rep.unit, 1 .- rep.unit))
        near = free[findall(u -> min(u, 1 - u) <= bound_tol, rep.unit)]
        record = get(store.evaluations, point_key(rep.unit), nothing)
        moments = record === nothing ? nothing : record.moments
        fit = objective === nothing || moments === nothing ? nothing :
              fit_summary(objective, moments; names=moment_names)
        LocalMinimum(rep.x, rep.unit, rep.fun, [r.index for r in group], count(r -> r.converged, group),
                     distance, near, moments, fit)
    end
    return SearchSummary(minima, labels, n_attempts(store), n_failed(store), length(completed),
                         count(r -> r.converged, completed), count(r -> r.fun === nothing, completed),
                         count(row -> row.status === :pending, rows), messages)
end

search_summary(result::TikTakResult; kwargs...) = search_summary(result.run_dir; kwargs...)

function Base.show(io::IO, ::MIME"text/plain", s::SearchSummary)
    print(io, "SearchSummary: ", length(s.minima), " distinct minima from ", s.n_completed,
          " completed local searches (", s.n_converged, " converged, ", s.n_infeasible, " infeasible, ",
          s.n_incomplete, " incomplete); ", s.n_evals, " evaluations, ", s.n_failed, " failed")
    rows = [["#", "loss", "found", "converged", "bound distance", "near bound", "x"]]
    for (rank, m) in enumerate(s.minima)
        push!(rows, [string(rank), _short(m.fun), string(length(m.searches)), string(m.n_converged),
                     _short(m.boundary_distance), join(s.parameter_names[m.near_bound], ", "),
                     "[" * join(_short.(m.x), ", ") * "]"])
    end
    _print_rows(io, rows; left=(6, 7))
end

Base.show(io::IO, s::SearchSummary) =
    print(io, "SearchSummary(", length(s.minima), " minima, ", s.n_completed, " completed searches)")
