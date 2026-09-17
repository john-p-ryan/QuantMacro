using Test
using Spline

include("helpers.jl")

@testset verbose = true "Spline" begin
    include("test_cubic.jl")
    include("test_hyman.jl")
    include("test_linear.jl")
    include("test_pchip.jl")
    include("test_ad.jl")
    include("test_bilinear.jl")
    include("test_multilinear.jl")
    include("test_pchip2d.jl")
    include("test_grid.jl")
end
