module TikTakPlotsExt

using TikTak
using Plots

function plot_binscatter(data; kwargs...)
    n_moments, n_parameters = size(data.panels)
    options = (; legend=false, size=(300 * n_parameters, 220 * n_moments), kwargs...,
               layout=(n_moments, n_parameters))
    panel = plot(; options...)
    # Plots numbers subplots across each row, whereas Julia matrices are column-major.
    for i in 1:n_moments, k in 1:n_parameters
        binned = data.panels[i, k]
        scatter!(panel, binned.x, binned.y; options..., subplot=(i - 1) * n_parameters + k,
                 xlabel=data.parameter_names[k], ylabel=data.moment_names[i], label="")
    end
    return panel
end

__init__() = (TikTak.BINSCATTER_BACKEND[] = plot_binscatter)

end # module
