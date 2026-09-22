@testset "local search" begin
    @testset "infeasible blend falls back to seed" begin
        dir = mktempdir()
        calls = Float64[]
        function holed(x)
            push!(calls, x[1])
            0.3 < x[1] < 0.7 && throw(ModelEvaluationError("hole at mixed start"))
            return (x[1] - 0.9)^2
        end
        store = TikTak.Store(dir)
        TikTak.initialize!(store, Dict{String,Any}(); resume=false, max_evals=100, deadline=nothing)
        transform = BoxTransform([(0, 1)])
        config = TikTakConfig(local_method=PatternSearchLocal(), local_max_evals=50)
        TikTak.evaluate_point(holed, transform, store, [0.9], "seed", config)
        TikTak.create_local!(store, 1, [0.5], [0.9])
        result = TikTak.run_local(holed, transform, store, 1, config)
        @test calls[1:2] ≈ [0.9, 0.5]
        @test result.fun == 0 && result.x == [0.9] && result.unit == [0.9]
        @test count(==(0.9), calls) == 1
        @test TikTak.local_row(store, 1).status == :done
        close(store)
    end

    @testset "nelder_mead and pattern_search on the unit box" begin
        f = u -> (u[1] - 0.3)^2 + (u[2] - 0.9)^2
        converged, message = TikTak.nelder_mead(f, [0.5, 0.5]; initial_step=0.1, x_tol=1e-7, f_tol=1e-12, max_calls=1000)
        @test converged && occursin("tolerance", message)
        converged, message = TikTak.nelder_mead(f, [0.5, 0.5]; initial_step=0.1, x_tol=1e-7, f_tol=1e-12, max_calls=5)
        @test !converged && occursin("limit", message)

        best = Ref((Inf, Float64[]))
        tracked(g) = u -> (v = g(u); v < best[][1] && (best[] = (v, copy(u))); v)
        boundary = u -> (u[1] + 1)^2 + (u[2] - 2)^2
        best[] = (Inf, Float64[])
        TikTak.nelder_mead(tracked(boundary), [0.5, 0.5]; initial_step=0.1, x_tol=1e-7, f_tol=1e-12, max_calls=1000)
        @test best[][2] ≈ [0, 1] atol=1e-6
        best[] = (Inf, Float64[])
        TikTak.pattern_search(tracked(boundary), [0.5, 0.5]; step=0.1, tolerance=1e-7, improvement_tol=0.0, max_calls=1000)
        @test best[][2] ≈ [0, 1] atol=1e-6

        converged, _ = TikTak.nelder_mead(u -> (u[1] - 0.25)^2, [0.9]; initial_step=0.1, x_tol=1e-7, f_tol=1e-12, max_calls=1000)
        @test converged
        holes = u -> u[1] < 0.5 ? Inf : (u[1] - 0.6)^2 + (u[2] - 0.6)^2
        best[] = (Inf, Float64[])
        TikTak.nelder_mead(tracked(holes), [0.55, 0.9]; initial_step=0.1, x_tol=1e-7, f_tol=1e-12, max_calls=1000)
        @test best[][2] ≈ [0.6, 0.6] atol=1e-5
        @test_throws ArgumentError TikTak.nelder_mead(f, Float64[]; initial_step=0.1, x_tol=1e-7, f_tol=1e-12, max_calls=10)
    end

    @testset "evaluator validation" begin
        dir = mktempdir()
        store = TikTak.Store(dir)
        TikTak.initialize!(store, Dict{String,Any}(); resume=false, max_evals=100, deadline=nothing)
        transform = BoxTransform([(0, 1)])
        config = TikTakConfig()
        @test_throws "outside the unit box" TikTak.Evaluator(x -> 1.0, transform, store, "t", config)([1.5])
        @test_throws ArgumentError TikTak.Evaluator(x -> [1.0], transform, store, "t", config)([0.5])
        @test TikTak.Evaluator(x -> 2, transform, store, "t", config)([0.25]) === 2.0
        @test TikTak.Evaluator(x -> Evaluation(1.0; residuals=[NaN]), transform, store, "t", config)([0.75]) == Inf
        @test TikTak.Evaluator(x -> throw(DomainError(1)), transform, store, "t",
                               TikTakConfig(failure_exceptions=(ModelEvaluationError, DomainError)))([0.1]) == Inf
        @test_throws DomainError TikTak.Evaluator(x -> throw(DomainError(1)), transform, store, "t", config)([0.2])
        close(store)
    end
end
