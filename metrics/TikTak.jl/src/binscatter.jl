# Binned moment associations across saved search evaluations, in physical units.

const BINSCATTER_BACKEND = Ref{Any}(nothing)

_binscatter_rows(source::Union{TikTakResult,AbstractString}, stage) =
    load_evaluations(source; status=:ok, stage=stage)
_binscatter_rows(source::AbstractVector{SavedEvaluation}, stage) =
    _evaluation_snapshot(source; status=:ok, stage=stage)

"""
    binscatter_data(source; nbins=20, stage=:full, parameters=nothing,
                    parameter_names=nothing, moment_names=nothing)

Bin saved model moments against each parameter, without evaluating the model.
`source` is a [`TikTakResult`](@ref), a run directory, or a vector of
[`SavedEvaluation`](@ref)s. Only successful evaluations with finite parameters
and moments are used; missing moments are skipped. `stage` is `:full` (default)
or `:screening`; the two objectives cannot be pooled.

Returns a named tuple with `panels`, `parameter_names`, `moment_names`,
`parameters`, and `n_evaluations`. `panels[i, k]` contains vectors `x`, `y`,
and `count`: mean parameter, mean raw moment, and number of evaluations in each
bin. Rows are moments and columns are parameters, as in a moment Jacobian.
`parameters` selects parameter indices in column order (default: all), and
names must cover all original parameters/moments. Use `parameters=jac.free`
to match the columns of a [`MomentJacobian`](@ref) with fixed parameters omitted.

Bins target equal counts using sorted parameter ranks, keeping tied values
together. Thus there can be fewer than `nbins` bins and unequal bin counts.
A constant parameter produces one bin. Every retained evaluation has equal
weight, including the optimizer's concentration of points near local minima.
These are marginal associations across search points, not partial derivatives:
other parameters vary within bins too.
"""
function binscatter_data(source; nbins::Integer=20, stage::Symbol=:full,
                         parameters=nothing, parameter_names=nothing, moment_names=nothing)
    nbins > 0 || throw(ArgumentError("nbins must be positive"))
    stage in (:full, :screening) || throw(ArgumentError("stage must be :full or :screening"))
    rows = filter(_binscatter_rows(source, stage)) do r
        r.moments !== nothing && !isempty(r.moments) &&
            all(isfinite, r.parameters) && all(isfinite, r.moments)
    end
    isempty(rows) && throw(ArgumentError("no successful finite moment evaluations in stage $stage"))
    n_parameters, n_moments = length(rows[1].parameters), length(rows[1].moments)
    all(r -> length(r.parameters) == n_parameters && length(r.moments) == n_moments, rows) ||
        throw(ArgumentError("evaluations must have consistent parameter and moment lengths"))
    selected = parameters === nothing ? collect(1:n_parameters) : collect(parameters)
    (!isempty(selected) && all(j -> j isa Integer && 1 <= j <= n_parameters, selected) &&
     allunique(selected)) || throw(ArgumentError("parameters must be distinct valid parameter indices"))
    selected = Int.(selected)
    pnames = _labels(parameter_names, n_parameters, "θ", "parameter_names")[selected]
    mnames = _labels(moment_names, n_moments, "m", "moment_names")
    panel_type = @NamedTuple{x::Vector{Float64}, y::Vector{Float64}, count::Vector{Int}}
    panels = Matrix{panel_type}(undef, n_moments, length(selected))
    n = length(rows)
    n_bins = min(nbins, n)
    for (k, j) in enumerate(selected)
        order = sortperm([r.parameters[j] for r in rows])
        sorted = [rows[t].parameters[j] for t in order]
        # Shift each quantile boundary right to avoid splitting identical x's.
        stops = unique!([searchsortedlast(sorted, sorted[cld(b * n, n_bins)]) for b in 1:n_bins])
        counts = diff([0; stops])
        x = zeros(length(stops))
        y = zeros(n_moments, length(stops))
        start = 1
        for (b, stop) in enumerate(stops)
            for t in start:stop
                row = rows[order[t]]
                x[b] += row.parameters[j] / counts[b]
                for i in 1:n_moments
                    y[i, b] += row.moments[i] / counts[b]
                end
            end
            start = stop + 1
        end
        for i in 1:n_moments
            panels[i, k] = (x=x, y=y[i, :], count=counts)
        end
    end
    return (; panels, parameter_names=pnames, moment_names=mnames,
            parameters=selected, n_evaluations=n)
end

"""
    binscatter(source; nbins=20, stage=:full, parameters=nothing,
               parameter_names=nothing, moment_names=nothing, kwargs...)

Return a Plots.jl scatter panel of saved moments (rows) against parameters
(columns), in original units. Install Plots and run `using Plots` first.
The returned plot supports `display` and `savefig`. See [`binscatter_data`](@ref)
for data selection, binning, and interpretation. Extra keywords such as `size`,
`markersize`, and `color` are passed to Plots; the matrix layout is fixed.
"""
function binscatter(source; nbins::Integer=20, stage::Symbol=:full, parameters=nothing,
                    parameter_names=nothing, moment_names=nothing, kwargs...)
    backend = BINSCATTER_BACKEND[]
    backend === nothing && throw(ArgumentError("binscatter requires Plots; install it and run `using Plots` first"))
    data = binscatter_data(source; nbins, stage, parameters, parameter_names, moment_names)
    return backend(data; kwargs...)
end
