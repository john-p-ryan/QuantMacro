@testset "minimize (serial)" begin
    @testset "local backends find the quadratic: $(m)" for m in METHODS
        dir = mktempdir()
        config = TikTakConfig(n_samples=16, n_local=4, local_max_evals=300, max_evals=1500,
                              local_method=m, x_tol=1e-6, f_tol=1e-12)
        result = minimize(quadratic, BOX2; config=config, run_dir=dir, problem_id="quadratic-v1", workers=Int[])
        @test has_solution(result) && result.status == :completed
        @test result.fun < 1e-8
        @test result.x ≈ [0.2, -0.4] atol=1e-4
        @test result.n_evals <= config.max_evals
        @test length(result.local_results) == 4 && sort([r.index for r in result.local_results]) == 1:4
        saved = JSON.parsefile(joinpath(dir, "result.json"))
        @test saved["fun"] == result.fun && saved["status"] == "completed"
    end

    @testset "rastrigin multimodal" begin
        result = minimize(rastrigin, [(-5.12, 5.12), (-5.12, 5.12)];
                          config=TikTakConfig(n_samples=256, n_local=32, local_max_evals=180, seed=7, x_tol=1e-6),
                          workers=Int[])
        @test result.fun < 1e-7
        @test result.x ≈ [0, 0] atol=1e-3
    end

    @testset "moments with unbounded parameter" begin
        truth = [0.65, 0.2]
        target = income_moments(truth)
        objective = MomentObjective(income_moments, target; scales=target)
        dir = mktempdir()
        result = minimize(objective, [(0, 0.98), (0, Inf)]; scale=[1, 0.2],
                          config=TikTakConfig(n_samples=64, n_local=8, local_max_evals=250, x_tol=1e-7),
                          run_dir=dir, problem_id="income-v1", workers=Int[])
        @test result.fun < 1e-9
        @test result.x ≈ truth atol=1e-5
        @test result.moments ≈ target atol=1e-6
        @test result.fun ≈ dot(result.residuals, result.residuals)
        ok = [r for r in values(history(dir).evaluations) if r.status == :ok]
        @test !isempty(ok) && all(r -> r.moments !== nothing && r.residuals !== nothing, ok)
    end

    @testset "rough surface and expected failures" begin
        result = minimize(rough_holes, [(-1, 1), (-1, 1)];
                          config=TikTakConfig(n_samples=128, n_local=16, local_method=PatternSearchLocal(),
                                              local_max_evals=150, x_tol=1e-4), workers=Int[])
        @test result.fun < 0.002
        @test result.n_failed > 0
        @test has_solution(result)
    end

    @testset "unbounded optimum and fixed parameter" begin
        objective = x -> (@assert x[3] == 3; (x[1] + 1.5)^2 + (x[2] - 2)^2)
        result = minimize(objective, [(-Inf, Inf), (-Inf, 5), (3, 3)];
                          config=TikTakConfig(n_samples=32, n_local=5, local_max_evals=200), workers=Int[])
        @test result.fun < 1e-7
        @test result.x[3] == 3
    end

    @testset "all fixed parameters evaluated once" begin
        dir = mktempdir()
        result = minimize(x -> sum(x .^ 2), [(2, 2), (3, 3)]; run_dir=dir, problem_id="fixed", workers=Int[])
        @test result.fun == 13 && result.x == [2, 3]
        @test result.n_evals == 1 && result.n_local_completed == 0
        @test result.status == :completed
    end

    @testset "warm start reevaluates and rejects stale resume" begin
        first_dir, second_dir = mktempdir(), mktempdir()
        config = TikTakConfig(n_samples=8, n_local=2)
        first = minimize(quadratic, BOX2; config=config, run_dir=first_dir, problem_id="v1", workers=Int[])
        warm = load_estimates(first_dir; limit=1)
        @test warm[1] == first.x
        second = minimize(x -> quadratic(x) + 5, BOX2; config=config, run_dir=second_dir, problem_id="v2",
                          warm_start=warm, workers=Int[])
        @test 5 <= second.fun < 5 + 1e-8
        @test_throws "specification" minimize(quadratic, BOX2; config=config, run_dir=first_dir,
                                              problem_id="v2", resume=true, workers=Int[])
        @test_throws "already exists" minimize(quadratic, BOX2; config=config, run_dir=first_dir,
                                               problem_id="v1", workers=Int[])
        @test_throws "no initialized run" minimize(quadratic, BOX2; config=config, run_dir=mktempdir(),
                                                   problem_id="v1", resume=true, workers=Int[])
        @test_throws ArgumentError minimize(quadratic, BOX2; config=config, run_dir=first_dir,
                                            problem_id="v1", resume=true, warm_start=warm, workers=Int[])
        matrix = minimize(quadratic, BOX2; config=config, warm_start=[0.2 -0.4; 0.0 0.0], workers=Int[])
        @test matrix.fun < 1e-20   # the unit-box round trip costs a few ulps
        @test matrix.x ≈ [0.2, -0.4] atol=1e-12
        @test_throws ArgumentError minimize(quadratic, BOX2; config=config, warm_start=[[1.0]], workers=Int[])
    end

    @testset "unexpected errors propagate and are recorded" begin
        dir = mktempdir()
        broken = x -> throw(ErrorException("programming bug"))
        @test_throws "programming bug" minimize(broken, [(0, 1)]; run_dir=dir, problem_id="broken", workers=Int[])
        records = collect(values(history(dir).evaluations))
        @test length(records) == 1 && records[1].status == :error
        @test occursin("programming bug", records[1].error)
        # Cached unexpected errors are not silently reused on resume.
        @test_throws "cached unexpected model error" minimize(broken, [(0, 1)]; run_dir=dir, problem_id="broken",
                                                              resume=true, workers=Int[])
        # The lock was released despite the error.
        @test_throws "already exists" minimize(broken, [(0, 1)]; run_dir=dir, problem_id="broken", workers=Int[])
    end

    @testset "no feasible solution and soft deadline" begin
        result = minimize(x -> Inf, [(0, 1)]; config=TikTakConfig(n_samples=8),
                          run_dir=joinpath(mktempdir(), "invalid"), problem_id="invalid", workers=Int[])
        @test !has_solution(result) && result.status == :no_feasible_point
        @test result.n_failed == 8 && result.n_local_completed == 0 && result.x === nothing
        slow = x -> (sleep(0.04); 1.0)
        timed = minimize(slow, [(0, 1)]; config=TikTakConfig(n_samples=8, max_seconds=0.025),
                         run_dir=joinpath(mktempdir(), "timed"), problem_id="timed", workers=Int[])
        @test timed.n_evals <= 1
        @test timed.status in (:budget_exhausted, :no_feasible_point)
    end

    @testset "invalid minimize inputs" begin
        @test_throws ArgumentError minimize(quadratic, BOX2; run_dir=mktempdir(), workers=Int[])
        @test_throws ArgumentError minimize(quadratic, BOX2; run_dir=mktempdir(), problem_id=" ", workers=Int[])
        @test_throws ArgumentError minimize(quadratic, BOX2; resume=true, workers=Int[])
    end

    @testset "serial TikTak mixes ranked seeds with the best completed local" begin
        dir = mktempdir()
        config = TikTakConfig(n_samples=16, n_local=4, local_max_evals=80)
        result = minimize(rastrigin, [(-5, 5), (-5, 5)]; config=config, run_dir=dir, problem_id="mixing", workers=Int[])
        store = history(dir)
        seeds = [Vector{Float64}(s) for s in get(store, "seeds")]
        starts = [r.start for r in TikTak.local_rows(store)]
        @test length(seeds) == length(starts) == 4
        @test starts[1] == seeds[1]
        for i in 2:length(seeds)
            done = result.local_results[1:i-1]
            best = done[argmin([(r.fun, r.index) for r in done])]
            weight = clamp(sqrt(i / length(seeds)), 0.1, 0.995)
            @test starts[i] ≈ (1 - weight) .* seeds[i] .+ weight .* best.unit
        end
        # Seeds are the best screening points in order.
        points = [Vector{Float64}(p) for p in get(store, "points")]
        screened = sort([(store.evaluations[TikTak.point_key(p)].value, i) for (i, p) in enumerate(points)])
        @test seeds == [points[i] for (_, i) in screened[1:4]]
    end

    @testset "interrupt is retryable and retains budget charge" begin
        dir = mktempdir()
        interrupted = Ref(false)
        function objective(x)
            if !interrupted[]
                interrupted[] = true
                throw(InterruptException())
            end
            return dot(x, x)
        end
        config = TikTakConfig(n_samples=4, n_local=1, local_max_evals=30)
        @test_throws InterruptException minimize(objective, [(-1, 1)]; config=config, run_dir=dir,
                                                 problem_id="interrupt", workers=Int[])
        @test only(values(history(dir).evaluations)).status == :abandoned
        result = minimize(objective, [(-1, 1)]; config=config, run_dir=dir, problem_id="interrupt",
                          resume=true, workers=Int[])
        unique_points = length(history(dir).evaluations)
        @test has_solution(result) && result.n_evals == unique_points + 1
        @test result.n_failed == 0
    end

    @testset "boundary optimum and warm seed: $(m)" for m in METHODS
        objective = x -> (@assert all(0 .<= x .<= 1); (x[1] + 1)^2 + (x[2] - 2)^2)
        result = minimize(objective, [(0, 1), (0, 1)]; warm_start=[[0.0, 1.0]],
                          config=TikTakConfig(n_samples=8, n_local=2, local_method=m), workers=Int[])
        @test result.x == [0, 1]
        @test result.fun == 2
    end

    @testset "custom local method" begin
        called = Ref(0)
        custom = CustomLocal(; name="two-point") do f, start, config
            called[] += 1
            f(clamp.(start .+ 0.01, 0, 1))
            return true, "custom done"
        end
        result = minimize(quadratic, BOX2; config=TikTakConfig(n_samples=8, n_local=3, local_method=custom), workers=Int[])
        @test called[] == 3 && has_solution(result)
        @test all(r -> r.converged && r.message == "custom done", result.local_results)
    end

    @testset "result show and to_dict" begin
        result = minimize(quadratic, BOX2; config=TikTakConfig(n_samples=8, n_local=1), workers=Int[])
        text = sprint(show, result)
        @test occursin("TikTakResult(status=:completed", text)
        d = TikTak.to_dict(result)
        @test d["status"] == "completed" && d["fun"] == result.fun && length(d["local_results"]) == 1
    end
end
