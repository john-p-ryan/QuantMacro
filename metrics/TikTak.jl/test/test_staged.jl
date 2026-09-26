@testset "staged accuracy" begin
    # Coarse stand-in: minimum shifted away from the fine one and lower values,
    # as a coarse grid biases rather than merely perturbs the criterion.
    coarse_quadratic(x) = sum((x .- [0.25, -0.35]) .^ 2) - 1.0

    @testset "screening uses the cheap objective; locals and the estimate use the full one" begin
        dir = mktempdir()
        coarse_calls, fine_calls = Vector{Float64}[], Vector{Float64}[]
        config = TikTakConfig(n_samples=16, n_local=3, local_max_evals=150)
        result = minimize(x -> (push!(fine_calls, copy(x)); quadratic(x)), BOX2; config=config,
                          screen_objective=x -> (push!(coarse_calls, copy(x)); coarse_quadratic(x)),
                          run_dir=dir, problem_id="staged-v1", workers=Int[])
        @test result.status == :completed && result.fun < 1e-8
        @test length(coarse_calls) == 16
        @test result.n_evals == length(coarse_calls) + length(fine_calls)
        store = history(dir)
        claims = [r for r in journal(dir) if r["t"] == "claim"]
        @test all(r -> get(r, "screening", false) == startswith(r["task"], "screen:"), claims)
        # The coarse values are lower, but never become the estimate.
        @test !store.evaluations[store.best_key].screening
        @test all(r -> r.fun >= 0, result.local_results)
        # Seeds are ranked by screening values.
        points = [Vector{Float64}(p) for p in get(store, "points")]
        screened = sort([(store.evaluations[TikTak.point_key(p; screening=true)].value, i)
                         for (i, p) in enumerate(points)])
        @test [Vector{Float64}(s) for s in get(store, "seeds")] == [points[i] for (_, i) in screened[1:3]]
        # The first local starts at the best seed: its screening value is not reused.
        seed_x = to_parameters(BoxTransform(BOX2), points[screened[1][2]])
        @test any(≈(seed_x), fine_calls)
        @test load_estimates(dir; limit=1)[1] == result.x
        @test length(load_estimates(dir; limit=10_000)) == count(r -> r.status === :ok, values(store.evaluations))
    end

    @testset "resume requires the same stages" begin
        dir = mktempdir()
        config = TikTakConfig(n_samples=16, n_local=3, max_evals=20)
        first = minimize(quadratic, BOX2; config=config, screen_objective=coarse_quadratic,
                         run_dir=dir, problem_id="staged-resume", workers=Int[])
        @test first.status == :budget_exhausted
        more = TikTakConfig(config; max_evals=600)
        @test_throws "specification differs" minimize(quadratic, BOX2; config=more, run_dir=dir,
                                                      problem_id="staged-resume", resume=true, workers=Int[])
        second = minimize(quadratic, BOX2; config=more, screen_objective=coarse_quadratic, run_dir=dir,
                          problem_id="staged-resume", resume=true, workers=Int[])
        @test second.status == :completed && second.fun < 1e-8
        claims = [r["key"] for r in journal(dir) if r["t"] == "claim"]
        @test length(claims) == length(unique(claims)) == second.n_evals
    end

    @testset "moment screening criterion is validated on resume" begin
        dir = mktempdir()
        target = income_moments([0.6, 0.3])
        fine = MomentObjective(income_moments, target; scales=target)
        config = TikTakConfig(n_samples=8, n_local=2, max_evals=12)
        minimize(fine, [(0, 0.95), (0.05, 1)]; config=config, run_dir=dir, problem_id="staged-moments",
                 screen_objective=MomentObjective(income_moments, target), workers=Int[])
        @test_throws "specification differs" minimize(fine, [(0, 0.95), (0.05, 1)]; config=config, run_dir=dir,
                                                      problem_id="staged-moments", resume=true, workers=Int[],
                                                      screen_objective=MomentObjective(income_moments, 2 .* target))
    end

    @testset "budget spent in screening is not reported as infeasible" begin
        result = minimize(quadratic, BOX2; config=TikTakConfig(n_samples=16, max_evals=16),
                          screen_objective=coarse_quadratic, workers=Int[])
        @test result.status == :budget_exhausted && !has_solution(result)
        failing = minimize(x -> Inf, BOX2; config=TikTakConfig(n_samples=8, n_local=2),
                           screen_objective=coarse_quadratic, workers=Int[])
        @test failing.status == :no_feasible_point
    end

    @testset "no free parameter evaluates the full objective" begin
        result = minimize(x -> 3.0, [(1, 1)]; screen_objective=x -> error("screening a fixed point"),
                          workers=Int[])
        @test result.fun == 3.0
    end

    @testset "distributed staged run" begin
        result = minimize(quadratic, BOX2; screen_objective=x -> quadratic(x) + 0.5,
                          config=TikTakConfig(n_samples=16, n_local=4, max_evals=700), workers=workers())
        @test result.status == :completed && result.fun < 1e-8
    end
end
