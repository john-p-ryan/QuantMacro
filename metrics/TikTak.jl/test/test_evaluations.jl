@testset "evaluation retrieval and workspace storage" begin
    # Fieldwise comparison handles NaN losses and independently copied vectors.
    same_rows(a, b) = length(a) == length(b) && all(
        all(isequal(getfield(x, f), getfield(y, f)) for f in fieldnames(SavedEvaluation))
        for (x, y) in zip(a, b))

    @testset "paired moments, disk round trip, and independent copies" begin
        target = income_moments([0.6, 0.3])
        objective = MomentObjective(income_moments, target; scales=target)
        result = minimize(objective, [(0.0, 0.9), (0.1, 1.0)]; workers=Int[],
                          config=TikTakConfig(n_samples=8, n_local=2, max_evals=50),
                          store_evaluations=true)
        rows = result.evaluations
        @test rows isa Vector{SavedEvaluation}
        @test length(rows) == result.n_evals
        @test issorted([r.sequence for r in rows])
        @test length(unique(r.key for r in rows)) == length(rows)
        @test same_rows(rows, load_evaluations(result.run_dir))
        @test same_rows(rows, load_evaluations(result))
        for r in rows
            @test r.status === :ok
            @test r.moments ≈ income_moments(r.parameters)
            @test r.residuals ≈ (r.moments .- target) ./ target
            @test r.value ≈ sum(abs2, r.residuals)
        end
        good = load_evaluations(result; status=:ok, stage=:full)
        @test size(stack([r.parameters for r in good]; dims=1)) == (length(good), 2)
        @test size(stack([r.moments for r in good]; dims=1)) == (length(good), 3)
        good[1].parameters[1] = -999
        good[1].moments[1] = -999
        good[1].residuals[1] = -999
        good[1].unit[1] = -999
        @test same_rows(rows, load_evaluations(result.run_dir))
        rows[1].parameters[1] = -123
        @test load_evaluations(result.run_dir)[1].parameters[1] != -123
        @test !haskey(JSON.parsefile(joinpath(result.run_dir, "result.json")), "evaluations")
        # The retained snapshot remains usable without the journal.
        rm(result.run_dir; recursive=true)
        @test same_rows(rows, load_evaluations(result))
    end

    @testset "defaults, scalar objectives, validation, and failures" begin
        result = minimize(x -> x[1]^2, [(0.5, 0.5)]; workers=Int[])
        @test result.evaluations === nothing
        rows = load_evaluations(result)
        @test length(rows) == 1
        @test rows[1].parameters == [0.5]
        @test rows[1].unit == Float64[]
        @test rows[1].value == 0.25
        @test rows[1].moments === rows[1].residuals === nothing
        @test isempty(load_evaluations(result; status=:failed))
        @test isempty(load_evaluations(result; stage=:screening))
        @test_throws ArgumentError load_evaluations(mktempdir())
        @test_throws ArgumentError load_evaluations(result; status=:typo)
        @test_throws ArgumentError load_evaluations(result; stage=:typo)

        failed = minimize(x -> throw(ModelEvaluationError("no equilibrium")), [(1, 1)];
                          workers=Int[], store_evaluations=true)
        @test failed.status === :no_feasible_point
        @test isempty(load_evaluations(failed; status=:ok))
        row = only(load_evaluations(failed; status=:failed))
        @test isnan(row.value) && row.moments === nothing
        @test occursin("no equilibrium", row.error)
        @test same_rows(failed.evaluations, load_evaluations(failed.run_dir))

        dir = mktempdir()
        @test_throws "programming bug" minimize(x -> error("programming bug"), [(1, 1)];
                        workers=Int[], run_dir=dir, problem_id="error-history", store_evaluations=true)
        @test occursin("programming bug", only(load_evaluations(dir; status=:error)).error)
    end

    @testset "resume can enable storage and keeps previous points" begin
        dir = mktempdir()
        config = TikTakConfig(n_samples=8, n_local=2, max_evals=4)
        first = minimize(quadratic, BOX2; config, run_dir=dir, problem_id="history-resume", workers=Int[])
        initial = load_evaluations(first)
        second = minimize(quadratic, BOX2; config=TikTakConfig(config; max_evals=50),
                          run_dir=dir, problem_id="history-resume", resume=true,
                          workers=Int[], store_evaluations=true)
        @test same_rows(initial, second.evaluations[1:4])
        @test length(second.evaluations) == second.n_evals
        @test same_rows(second.evaluations, load_evaluations(dir))
        third = minimize(quadratic, BOX2; config=TikTakConfig(config; max_evals=75),
                         run_dir=dir, problem_id="history-resume", resume=true, workers=Int[])
        @test third.evaluations === nothing
        @test length(load_evaluations(second)) == second.n_evals
        @test length(load_evaluations(dir)) == third.n_evals
    end

    @testset "stages and callback snapshots across executors" begin
        for executor in (InlineExecutor(), ThreadedExecutor(2), DistributedExecutor(workers()))
            snapshots = TikTakResult[]
            result = minimize(x -> Evaluation(x[1]^2; moments=[x[1], x[1]^2]), [(-1, 1)];
                screen_objective=x -> Evaluation(x[1]^2 - 1; moments=[x[1]]), executor,
                config=TikTakConfig(n_samples=8, n_local=2, max_evals=50),
                store_evaluations=true, callback=r -> push!(snapshots, r))
            @test !isempty(snapshots)
            @test all(r -> r.evaluations !== nothing, snapshots)
            @test same_rows(result.evaluations, load_evaluations(result.run_dir))
            coarse = load_evaluations(result; stage=:screening, status=:ok)
            full = load_evaluations(result; stage=:full, status=:ok)
            @test length(coarse) == 8
            @test length(coarse) + length(full) == result.n_evals
            @test all(r -> length(r.moments) == 1 && r.value ≈ r.parameters[1]^2 - 1, coarse)
            @test all(r -> length(r.moments) == 2 && r.value ≈ r.parameters[1]^2, full)
            snapshots[1].evaluations[1].parameters[1] = -999
            @test result.evaluations[1].parameters[1] != -999
        end
    end

    @testset "pending, abandoned, and retried cache entries" begin
        dir = mktempdir()
        store = TikTak.Store(dir)
        try
            TikTak.initialize!(store, Dict{String,Any}(); resume=false, max_evals=5, deadline=nothing)
            _, key, _ = TikTak.claim!(store, [0.5], [2.0], "first", nothing)
            @test only(load_evaluations(dir; status=:pending)).parameters == [2.0]
            TikTak.abandon!(store, key)
            @test length(load_evaluations(dir; status=:abandoned)) == 1
            TikTak.claim!(store, [0.5], [2.0], "retry", nothing)
            TikTak.finish!(store, key; value=3.0, moments=[4.0])
            TikTak.claim!(store, [0.5], [2.0], "cache-hit", nothing)
            row = only(load_evaluations(dir))
            @test row.status === :ok && row.sequence == 2
            @test row.value == 3.0 && row.moments == [4.0]
            @test TikTak.n_attempts(store) == 2
        finally
            close(store)
        end
    end
end
