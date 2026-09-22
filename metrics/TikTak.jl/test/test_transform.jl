@testset "BoxTransform" begin
    @testset "roundtrip: fixed, finite, and infinite bounds" begin
        t = BoxTransform([(-2, 5), (1, Inf), (-Inf, 3), (-Inf, Inf), (7, 7)];
                         scale=[1, 2, 4, 3, 1], location=[0, 0, 0, -2, 0])
        @test t.dimension == 4 && t.size == 5 && t.free == [1, 2, 3, 4]
        for u in (zeros(4), ones(4), [0.1, 0.3, 0.8, 0.6])
            x = to_parameters(t, u)
            @test all(isfinite, x)
            @test x[end] == 7
            @test to_unit(t, x) ≈ u atol=1e-10
        end
        @test to_parameters(t, zeros(4))[1] == -2 && to_parameters(t, ones(4))[1] == 5
        @test_throws "cutoff" to_unit(t, [0, 1e20, 0, 0, 7])
        @test_throws "violate" to_unit(t, [0, 0, 0, 0, 7])
        @test_throws ArgumentError to_parameters(t, [0.5, 0.5, 0.5, 1.5])
        @test_throws ArgumentError to_parameters(t, [0.5, 0.5, 0.5])
    end

    @testset "invalid inputs" begin
        for bounds in ([(2, 1)], [(Inf, Inf)], [(NaN, 1)], [], [(-Inf, -Inf)])
            @test_throws ArgumentError BoxTransform(bounds)
        end
        @test_throws ArgumentError BoxTransform([(0, 1)]; tail=0.5)
        @test_throws ArgumentError BoxTransform([(0, 1)]; scale=0)
        @test_throws ArgumentError BoxTransform([(0, 1), (0, 1)]; scale=[1])
        @test_throws ArgumentError BoxTransform([(-Inf, Inf)]; location=NaN)
        @test_throws "overflow" BoxTransform([(0, Inf)]; scale=1e308)
    end

    @testset "input forms and specification" begin
        t1 = BoxTransform([-1 1; 0 2])
        t2 = BoxTransform([-1 => 1, 0 => 2])
        t3 = BoxTransform([[-1, 1], [0, 2]])
        @test t1.lower == t2.lower == t3.lower == [-1, 0]
        @test t1.upper == t2.upper == t3.upper == [1, 2]
        spec = TikTak.specification(BoxTransform([(0, Inf), (-Inf, 1)]; scale=2))
        @test spec["bounds"] == [[0.0, "Inf"], ["-Inf", 1.0]]
        @test spec["scale"] == [2.0, 2.0] && spec["tail"] == 1e-6
    end
end
