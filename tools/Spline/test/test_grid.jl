@testset "make_grid" begin
    g = make_grid(0.0, 1.0, 5)
    @test g == [0.0, 0.25, 0.5, 0.75, 1.0]
    @test make_grid(1.0, 3.0, 4) ≈ range(1.0, 3.0, length=4)
    # density > 1 concentrates points near x_min, density < 1 near x_max.
    @test make_grid(0.0, 1.0, 5; density=2.0) ≈ [0.0, 1 / 16, 1 / 4, 9 / 16, 1.0]
    for density in [0.5, 1.0, 2.0, 3.0]
        g = make_grid(-2.0, 5.0, 11; density=density)
        @test length(g) == 11
        @test g[1] == -2.0 && g[end] == 5.0
        @test issorted(g) && allunique(g)
        if density > 1
            @test all(diff(diff(g)) .> 0)
        elseif density < 1
            @test all(diff(diff(g)) .< 0)
        end
    end
end
