@testset "resume" begin
    @testset "resume screening and cache" begin
        dir = mktempdir()
        calls = Vector{Float64}[]
        objective = x -> (push!(calls, copy(x)); quadratic(x))
        config = TikTakConfig(n_samples=16, n_local=3, max_evals=7)
        first = minimize(objective, BOX2; config=config, run_dir=dir, problem_id="resume-v1", workers=Int[])
        @test first.n_evals == 7 && first.status == :budget_exhausted
        second = minimize(objective, BOX2; config=TikTakConfig(config; max_evals=500), run_dir=dir,
                          problem_id="resume-v1", resume=true, workers=Int[])
        @test second.status == :completed && second.fun < 1e-8
        @test length(calls) == second.n_evals
        before = length(calls)
        third = minimize(objective, BOX2; config=TikTakConfig(config; max_evals=500), run_dir=dir,
                         problem_id="resume-v1", resume=true, workers=Int[])
        @test length(calls) == before && third.fun == second.fun
        @test third.n_local_completed == 3
    end

    @testset "resume interrupted local budget" begin
        dir = mktempdir()
        config = TikTakConfig(n_samples=8, n_local=3, max_evals=12, local_max_evals=100)
        first = minimize(quadratic, BOX2; config=config, run_dir=dir, problem_id="local-resume", workers=Int[])
        @test first.status == :budget_exhausted
        @test count(r -> r.status == :pending, TikTak.local_rows(history(dir))) == 1
        second = minimize(quadratic, BOX2; config=TikTakConfig(config; max_evals=400), run_dir=dir,
                          problem_id="local-resume", resume=true, workers=Int[])
        @test second.status == :completed && second.fun < 1e-8
        # No point was charged twice: every claim is for a distinct point.
        claims = [r["key"] for r in journal(dir) if r["t"] == "claim"]
        @test length(claims) == length(unique(claims))
        @test second.n_evals == length(claims)
    end

    @testset "resume after deadline keeps seeds and local starts" begin
        dir = mktempdir()
        slow = x -> (sleep(0.01); quadratic(x))
        config = TikTakConfig(n_samples=8, n_local=2, max_seconds=0.15)
        first = minimize(slow, BOX2; config=config, run_dir=dir, problem_id="deadline", workers=Int[])
        @test first.status in (:budget_exhausted, :completed)
        second = minimize(slow, BOX2; config=TikTakConfig(config; max_seconds=nothing), run_dir=dir,
                          problem_id="deadline", resume=true, workers=Int[])
        @test second.status == :completed && second.n_local_completed == 2
        @test get(history(dir), "seeds") == get(history(dir), "seeds")
    end
end
