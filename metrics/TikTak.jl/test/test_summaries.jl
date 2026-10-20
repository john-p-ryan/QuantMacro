@testset "summaries" begin
    @testset "fit_summary" begin
        target = [1.0, 2.0, 3.0]
        moments = [1.5, 1.0, 3.5]
        diagonal = MomentObjective(identity, target; weights=[1.0, 2.0, 0.5], scales=[0.5, 1.0, 2.0])
        s = fit_summary(diagonal, moments)
        scaled = (moments .- target) ./ [0.5, 1.0, 2.0]
        @test s.names == ["m1", "m2", "m3"]
        @test s.errors == moments .- target && s.scaled_errors == scaled
        @test s.contributions == [1.0, 2.0, 0.5] .* scaled .^ 2
        @test s.loss ≈ diagonal(moments).value && sum(s.contributions) ≈ s.loss

        weights = [2.0 1.5 0.0; 1.5 2.0 0.0; 0.0 0.0 1.0]   # positive definite with strong cross terms
        full = MomentObjective(identity, target; weights=weights)
        t = fit_summary(full, [1.5, 1.8, 3.0]; names=["a", "b", "c"])
        @test t.names == ["a", "b", "c"]
        @test sum(t.contributions) ≈ full([1.5, 1.8, 3.0]).value ≈ t.loss
        @test t.contributions[2] < 0    # g = [0.5, -0.2, 0]: cross term outweighs b's own term

        @test_throws ArgumentError fit_summary(diagonal, [1.0, 2.0])
        @test_throws ArgumentError fit_summary(diagonal, moments; names=["a"])
    end

    @testset "rastrigin minima" begin
        dir = mktempdir()
        result = minimize(rastrigin, [(-5.12, 5.12), (-5.12, 5.12)];
                          config=TikTakConfig(n_samples=64, n_local=12, local_max_evals=40, seed=3),
                          run_dir=dir, problem_id="rastrigin-summary", workers=Int[])
        s = search_summary(dir; x_tol=1e-2)
        feasible = count(r -> r.fun !== nothing, result.local_results)
        @test sum(length(m.searches) for m in s.minima) == feasible
        @test s.n_completed == length(result.local_results) && s.n_incomplete == 0
        # Unconverged searches in one basin are rediscoveries, not distinct minima.
        @test length(s.minima) > 1 && length(s.minima[1].searches) > 1
        @test issorted([m.fun for m in s.minima])
        @test s.minima[1].fun == result.fun && s.minima[1].x == result.x
        @test s.n_evals == result.n_evals && s.n_failed == result.n_failed
        @test sum(values(s.messages)) == s.n_completed
        @test s.parameter_names == ["θ1", "θ2"]
        @test all(m.fit === nothing && m.moments === nothing for m in s.minima)
        for i in eachindex(s.minima), j in 1:i-1
            a, b = s.minima[j], s.minima[i]
            @test maximum(abs, a.unit - b.unit) > 1e-2
        end
        same = search_summary(result; x_tol=1e-2)
        @test [m.searches for m in same.minima] == [m.searches for m in s.minima]
        @test [m.x for m in same.minima] == [m.x for m in s.minima]
        @test_throws ArgumentError search_summary(mktempdir())
    end

    @testset "moment run attaches fit" begin
        target = income_moments([0.65, 0.2])
        objective = MomentObjective(income_moments, target; scales=target)
        dir = mktempdir()
        result = minimize(objective, [(0, 0.98), (0, Inf)]; scale=[1, 0.2],
                          config=TikTakConfig(n_samples=64, n_local=4, local_max_evals=250),
                          run_dir=dir, problem_id="income-summary", workers=Int[])
        s = search_summary(dir; moment_names=["var", "cov", "cov2"], parameter_names=["rho", "sigma"])
        best = s.minima[1]
        @test best.fit !== nothing && best.fit.names == ["var", "cov", "cov2"]
        @test best.fit.loss ≈ best.fun && best.fit.moments == best.moments == result.moments
        @test best.fit.target == target
        @test sprint(show, MIME"text/plain"(), best.fit) |> text -> occursin("var", text) && occursin("share", text)
        @test_throws ArgumentError search_summary(dir; parameter_names=["rho"])

        weighted = MomentObjective(income_moments, target; scales=target, weights=[2.0 0.5 0.0; 0.5 1.0 0.0; 0.0 0.0 1.0])
        wdir = mktempdir()
        minimize(weighted, [(0, 0.98), (0, Inf)]; scale=[1, 0.2],
                 config=TikTakConfig(n_samples=16, n_local=2, local_max_evals=60),
                 run_dir=wdir, problem_id="income-full", workers=Int[])
        for m in search_summary(wdir).minima
            @test m.fit.loss ≈ m.fun
        end
    end

    @testset "boundary minimum maps free indices" begin
        dir = mktempdir()
        objective = x -> (x[1] + 1)^2 + (x[2] - 2)^2
        minimize(objective, [(0, 1), (0, 1), (5, 5)]; config=TikTakConfig(n_samples=16, n_local=3),
                 run_dir=dir, problem_id="corner", workers=Int[])
        best = search_summary(dir; parameter_names=["a", "b", "c"]).minima[1]
        @test best.near_bound == [1, 2] && best.boundary_distance == 0
        @test best.x[3] == 5
        @test occursin("a, b", sprint(show, MIME"text/plain"(), search_summary(dir; parameter_names=["a", "b", "c"])))
    end

    @testset "incomplete searches are not clustered" begin
        dir = mktempdir()
        result = minimize(rastrigin, [(-5, 5), (-5, 5)];
                          config=TikTakConfig(n_samples=16, n_local=4, local_max_evals=100, max_evals=110),
                          run_dir=dir, problem_id="short", workers=Int[])
        s = search_summary(dir)
        @test result.status == :budget_exhausted
        @test s.n_incomplete > 0
        @test s.n_completed > 0 && sum(length(m.searches) for m in s.minima; init=0) <= s.n_completed
        rows = TikTak.local_rows(history(dir))
        done = [r.result for r in rows if r.status === :done && r.result !== nothing && r.result.fun !== nothing]
        @test sum(length, [m.searches for m in s.minima]; init=0) == length(done)
        @test s.n_incomplete == count(r -> r.status === :pending, rows)
        text = sprint(show, MIME"text/plain"(), s)
        @test occursin("incomplete", text) && occursin("distinct minima", text)
    end
end
