# Activates `NLoptLocal` when NLopt is loaded on the process.
module TikTakNLoptExt

using TikTak
using NLopt

function nlopt_search(method::NLoptLocal, f, start, config)
    dimension = length(start)
    opt = NLopt.Opt(method.algorithm, dimension)
    opt.lower_bounds = zeros(dimension)
    opt.upper_bounds = ones(dimension)
    opt.maxeval = config.local_max_evals
    opt.xtol_abs = config.x_tol
    opt.ftol_abs = config.f_tol
    opt.initial_step = fill(config.initial_step, dimension)

    # Explicit method options override the defaults from the TikTak configuration.
    for (key, value) in method.options
        setproperty!(opt, key, value)
    end
    # Exceptions thrown by `f` (including budget signals) force NLopt to stop and
    # are rethrown by `NLopt.optimize`.
    opt.min_objective = (x, grad) -> f(x)
    _, _, status = NLopt.optimize(opt, Vector{Float64}(start))
    converged = status in (:SUCCESS, :STOPVAL_REACHED, :FTOL_REACHED, :XTOL_REACHED)
    return converged, "NLopt $(method.algorithm): $status"
end

__init__() = (TikTak.NLOPT_BACKEND[] = nlopt_search)

end # module
