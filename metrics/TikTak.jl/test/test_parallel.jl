@testset "parallel execution" begin
    @testset "distributed workers and strict budget" begin
        dir = mktempdir()
        # Default worker selection: every process added with addprocs.
        result = minimize(process_objective, [(-1, 1), (-1, 1)];
                          config=TikTakConfig(n_samples=64, n_local=4, max_evals=13),
                          run_dir=dir, problem_id="process-v1")
        @test result.n_evals == 13
        @test result.status == :budget_exhausted
        pids = Set(Int(r.moments[1]) for r in values(history(dir).evaluations) if r.status == :ok)
        @test pids == Set(workers())
        @test !(myid() in pids)
    end

    @testset "parallel locals complete" begin
        result = minimize(quadratic, BOX2; config=TikTakConfig(n_samples=16, n_local=6, max_evals=700), workers=workers())
        @test result.n_local_completed == 6 && result.fun < 1e-8
        @test sort([r.index for r in result.local_results]) == 1:6
    end

    @testset "distributed local backends: $(m)" for m in (NelderMeadLocal(), PatternSearchLocal(), NLoptLocal(:LN_BOBYQA))
        result = minimize(quadratic, BOX2; workers=workers(),
                          config=TikTakConfig(n_samples=16, n_local=4, local_method=m, x_tol=1e-6, f_tol=1e-12))
        @test result.fun < 1e-8
    end

    @testset "threaded executor" begin
        result = minimize(quadratic, BOX2; config=TikTakConfig(n_samples=16, n_local=6, max_evals=700),
                          executor=ThreadedExecutor(3))
        @test result.n_local_completed == 6 && result.fun < 1e-8
    end

    @testset "callback and target value" begin
        seen = Symbol[]
        result = minimize(quadratic, BOX2; config=TikTakConfig(n_samples=16, n_local=4, target_value=1e-6),
                          workers=workers(), callback=r -> push!(seen, r.status))
        @test result.status == :target_reached
        @test result.n_local_completed >= 1 && result.fun <= 1e-6
        @test !isempty(seen) && all(==(:running), seen)
    end

    @testset "worker errors propagate and are recorded" begin
        dir = mktempdir()
        @test_throws RemoteException minimize(broken_remote, [(0, 1)]; run_dir=dir, problem_id="broken", workers=workers())
        # Both workers may have hit the error before the coordinator aborted.
        records = collect(values(history(dir).evaluations))
        @test !isempty(records) && all(r -> r.status == :error, records)
        # The run context was removed from the workers and the store unregistered.
        @test all(pid -> remotecall_fetch(() -> isempty(TikTak.CONTEXTS), pid), workers())
        @test isempty(TikTak.ACTIVE_STORES)
    end

    @testset "moments across workers with resume" begin
        dir = mktempdir()
        target = income_moments([0.65, 0.2])
        objective = MomentObjective(income_moments, target; scales=target)
        config = TikTakConfig(n_samples=32, n_local=4, local_max_evals=150, max_evals=60)
        first = minimize(objective, [(0, 0.98), (0, Inf)]; scale=[1, 0.2], config=config,
                         run_dir=dir, problem_id="income-parallel", workers=workers())
        @test first.status == :budget_exhausted && first.n_evals == 60
        second = minimize(objective, [(0, 0.98), (0, Inf)]; scale=[1, 0.2], config=TikTakConfig(config; max_evals=2000),
                          run_dir=dir, problem_id="income-parallel", resume=true, workers=workers())
        @test second.status == :completed && second.fun < 1e-8
        @test second.x ≈ [0.65, 0.2] atol=1e-4
    end

    @testset "executor validation" begin
        @test_throws ArgumentError DistributedExecutor([myid()])
        @test_throws ArgumentError DistributedExecutor([999])
        @test_throws ArgumentError DistributedExecutor(Int[])
        @test_throws ArgumentError ThreadedExecutor(0)
        @test_throws ArgumentError minimize(quadratic, BOX2; workers=workers(), executor=InlineExecutor())
        @test_throws ArgumentError minimize(quadratic, BOX2; executor=:threads)
        @test TikTak.capacity(DistributedExecutor(workers())) == length(workers())
        @test TikTak.capacity(InlineExecutor()) == 1
    end
end
