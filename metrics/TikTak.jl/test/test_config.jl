@testset "TikTakConfig" begin
    @testset "validation" begin
        @test_throws ArgumentError TikTakConfig(mixing_max=1)
        @test_throws ArgumentError TikTakConfig(x_tol=0.2, initial_step=0.1)
        @test_throws ArgumentError TikTakConfig(initial_step=0.6)
        @test_throws ArgumentError TikTakConfig(n_samples=0)
        @test_throws ArgumentError TikTakConfig(n_local=0)
        @test_throws ArgumentError TikTakConfig(seed=-1)
        @test_throws ArgumentError TikTakConfig(failure_exceptions=(DomainError,))
        @test_throws ArgumentError TikTakConfig(failure_exceptions=(ModelEvaluationError, "x"))
        @test_throws ArgumentError TikTakConfig(local_method=:nelder_mead)
        @test_throws ArgumentError TikTakConfig(max_seconds=0)
        @test_throws ArgumentError TikTakConfig(target_value=Inf)
        @test_throws ArgumentError TikTakConfig(f_tol=-1)
        @test TikTakConfig(failure_exceptions=(ModelEvaluationError, DomainError)).failure_exceptions ==
              (ModelEvaluationError, DomainError)
    end

    @testset "copy with overrides and specification" begin
        base = TikTakConfig(n_samples=16, n_local=3)
        copied = TikTakConfig(base; max_evals=500)
        @test copied.n_samples == 16 && copied.n_local == 3 && copied.max_evals == 500
        spec = TikTak.specification(copied)
        @test !haskey(spec, "max_evals") && !haskey(spec, "max_seconds")
        @test spec["local_method"]["method"] == "NelderMeadLocal"
        @test spec["failure_exceptions"] == ["TikTak.ModelEvaluationError"]
        @test TikTak.specification(TikTakConfig(base; max_evals=1, max_seconds=5)) == spec
        @test TikTak.specification(TikTakConfig(base; seed=1)) != spec
        nlopt = TikTak.specification(NLoptLocal(:LN_SBPLX; xtol_rel=1e-8))
        @test nlopt["algorithm"] == "LN_SBPLX" && nlopt["options"]["xtol_rel"] == "1.0e-8"
        custom = CustomLocal((f, start, config) -> (f(start); (true, "done")); name="noop")
        @test TikTak.specification(custom)["name"] == "noop"
    end

    @testset "sobol points" begin
        points = sobol_points(3, 100, 0)
        @test length(points) == 100
        @test all(p -> length(p) == 3 && all(0 .<= p .< 1), points)
        @test length(unique(points)) == 100
        @test sobol_points(3, 100, 0) == points
        @test sobol_points(3, 100, 1) != points
        @test length(sobol_points(2, 1, 0)) == 1
        # Digital scrambling keeps the net balanced: every axis half holds half the points.
        block = sobol_points(4, 64, 3)
        for j in 1:4
            @test count(p -> p[j] < 0.5, block) == 32
            @test count(p -> p[j] < 0.25, block) == 16
        end
        @test_throws ArgumentError sobol_points(0, 4, 0)
    end
end
