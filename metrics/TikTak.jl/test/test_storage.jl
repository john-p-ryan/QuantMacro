@testset "storage" begin
    @testset "exact cache deduplicates concurrent calls" begin
        dir = mktempdir()
        calls = Threads.Atomic{Int}(0)
        objective = x -> (Threads.atomic_add!(calls, 1); sleep(0.1); sum(x .^ 2))
        store = TikTak.Store(dir)
        TikTak.initialize!(store, Dict{String,Any}(); resume=false, max_evals=10, deadline=nothing)
        transform, config = BoxTransform([(-1, 1)]), TikTakConfig()
        tasks = [(@async TikTak.Evaluator(objective, transform, store, "test:$i", config)([0.3])) for i in 1:4]
        values = fetch.(tasks)
        @test all(v -> v ≈ 0.16, values)
        @test calls[] == 1 == TikTak.n_attempts(store)
        @test length(TikTak.best(store, "test:4").unit) == 1
        close(store)
    end

    @testset "coordinator lock" begin
        dir = mktempdir()
        TikTak.coordinator_lock(dir) do
            @test_throws "coordinator" TikTak.coordinator_lock(() -> nothing, dir)
        end
        @test TikTak.coordinator_lock(() -> :released, dir) == :released
    end

    @testset "journal replay and load_estimates" begin
        dir = mktempdir()
        config = TikTakConfig(n_samples=8, n_local=2)
        result = minimize(quadratic, BOX2; config=config, run_dir=dir, problem_id="v1", workers=Int[])
        warm = load_estimates(dir; limit=3)
        @test warm[1] == result.x && length(warm) == 3
        @test_throws ArgumentError load_estimates(dir; limit=0)
        @test_throws ArgumentError load_estimates(mktempdir())
        store = history(dir)
        @test TikTak.n_attempts(store) == result.n_evals
        @test TikTak.best(store).value == result.fun
        @test length(TikTak.local_rows(store)) == 2
        # A truncated final line (crash mid-write) is ignored; earlier corruption is not.
        path = joinpath(dir, "history.jsonl")
        open(path, "a") do io
            write(io, "{\"t\":\"claim\",\"key\"")
        end
        @test TikTak.n_attempts(history(dir)) == result.n_evals
        open(path, "a") do io
            write(io, "\n{\"t\":\"meta\",\"k\":\"x\",\"v\":1}\n")
        end
        @test_throws "corrupt journal" history(dir)
    end

    @testset "point keys" begin
        @test TikTak.point_key([0.0, 1.0]) == TikTak.point_key([-0.0, 1.0])
        @test TikTak.point_key([0.1]) != TikTak.point_key([0.1 + 1e-16])
        @test TikTak.point_key(Float64[]) == TikTak.point_key(Float64[])
    end
end
