@testset "criterion profiles" begin
    # Linear moments under efficient weights: the profile of θ1 is exactly
    # (v - θ̂1)² / V₁₁ with V = (X'Ω⁻¹X)⁻¹, so the profile interval equals the Wald interval.
    X = [1.0 0.0; 0.0 1.0; 1.0 1.0; 1.0 -0.5]
    target = X * [0.3, -0.2] .+ [0.02, -0.01, 0.03, -0.02]
    Σ = [1.0 0.5 0.2 0.1; 0.5 2.0 0.3 0.0; 0.2 0.3 1.5 0.4; 0.1 0.0 0.4 0.8]
    nobs = 200
    Ω = Σ / nobs
    bounds = [(-1, 1), (-1, 1)]
    calls = Ref(0)
    linear(x) = (calls[] += 1; X * x)
    objective = MomentObjective(linear, target; weights=inverse_covariance_weights(Σ; nobs=nobs))
    V = inv(X' * inv(Ω) * X)
    θ̂ = V * X' * inv(Ω) * target
    se = sqrt(V[1, 1])
    config = TikTakConfig(n_samples=8, n_local=2, local_max_evals=400, x_tol=1e-8, f_tol=1e-14)
    grid = θ̂[1] .+ se .* [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
    saved_points(dir) = [Vector{Float64}(p) for p in get(history(dir), "points")]

    @testset "exact quadratic profile and interval" begin
        dir = mktempdir()
        p = profile(objective, bounds, θ̂, "a", grid; names=["a", "b"], config=config, run_dir=dir,
                    problem_id="linear-v1", workers=Int[])
        @test p.index == 1 && p.name == "a" && p.values == grid && isempty(p.issues)
        @test p.loss ≈ objective(θ̂).value && all(==(:completed), p.status)
        @test p.delta ≈ [9.0, 4.0, 1.0, 1.0, 4.0, 9.0] rtol=1e-6
        # The other parameter follows its conditional GLS solution.
        for (v, x) in zip(grid, p.x)
            @test x[1] == v
            @test x[2] ≈ θ̂[2] + V[2, 1] / V[1, 1] * (v - θ̂[1]) atol=1e-6
        end
        interval = profile_interval(p)
        @test interval.lower ≈ θ̂[1] - 1.959963984540054 * se rtol=1e-6
        @test interval.upper ≈ θ̂[1] + 1.959963984540054 * se rtol=1e-6
        @test interval.lower_bounded && interval.upper_bounded && !interval.disjoint
        narrow = profile_interval(p; critical=1.0)
        @test [narrow.lower, narrow.upper] ≈ θ̂[1] .+ [-se, se] rtol=1e-6
        wide = profile_interval(p; critical=16.0)
        @test wide.lower == grid[1] && wide.upper == grid[end] && !wide.lower_bounded && !wide.upper_bounded
        @test_throws "critical" profile_interval(p; critical=0)

        # Separate, versioned runs warm-started from the estimate and the neighbour.
        point_dir = joinpath(dir, "p1_$(grid[4])")
        @test p.results[4].run_dir == point_dir
        @test get(history(point_dir), "specification")["problem_id"] == "linear-v1/profile/p1=$(grid[4])"
        transform = BoxTransform([(grid[4], grid[4]), (-1, 1)])
        @test to_unit(transform, [grid[4], θ̂[2]]) in saved_points(point_dir)
        neighbour = joinpath(dir, "p1_$(grid[5])")
        @test to_unit(BoxTransform([(grid[5], grid[5]), (-1, 1)]), [grid[5], p.x[4][2]]) in saved_points(neighbour)
        text = sprint(show, MIME"text/plain"(), p)
        @test occursin("increase", text) && occursin(dir, text)

        # Resume reuses every saved point; a refined grid adds only the new ones.
        before = calls[]
        again = profile(objective, bounds, θ̂, 1, grid; config=config, run_dir=dir, problem_id="linear-v1",
                        resume=true, workers=Int[])
        @test calls[] == before + 1 && again.losses == p.losses   # only the loss at the estimate
        refined = profile(objective, bounds, θ̂, 1, [grid; θ̂[1] + 0.5se]; config=config, run_dir=dir,
                          problem_id="linear-v1", resume=true, workers=Int[])
        @test length(refined.values) == 7
        @test refined.delta[4] ≈ 0.25 rtol=1e-6
        @test_throws "already exists" profile(objective, bounds, θ̂, 1, grid; config=config, run_dir=dir,
                                              problem_id="linear-v1", workers=Int[])
        shifted = MomentObjective(x -> X * x .+ 0.01, target; weights=inverse_covariance_weights(Σ; nobs=nobs))
        @test_throws "the model, data, or draws changed" profile(shifted, bounds, θ̂, 1, grid; config=config,
                                                                 run_dir=dir, problem_id="linear-v1",
                                                                 resume=true, workers=Int[])
        changed = MomentObjective(objective; weights=inverse_covariance_weights(2Σ; nobs=nobs))
        # A new estimate skips the loss check; the saved specification still refuses new weights.
        @test_throws "differs" profile(changed, bounds, θ̂ .+ [0.0, 1e-3], 1, grid; config=config, run_dir=dir,
                                       problem_id="linear-v1", resume=true, workers=Int[])
    end

    @testset "user warm starts, targets, and nuisance parameters" begin
        # A target value would stop searches early, so profiles drop it.
        targeted = profile(objective, bounds, θ̂, 1, grid[[1, 6]]; config=TikTakConfig(config; target_value=1e10),
                           warm_start=[0.0, 0.9], workers=Int[])
        @test all(==(:completed), targeted.status)
        @test targeted.delta ≈ [9.0, 9.0] rtol=1e-6
        transform = BoxTransform([(grid[1], grid[1]), (-1, 1)])
        @test to_unit(transform, [grid[1], 0.9]) in saved_points(targeted.results[1].run_dir)
        # A fixed parameter and an unbounded nuisance parameter; the grid contains the estimate.
        three(x) = [X * x[1:2]; x[3]]
        fixed_objective = MomentObjective(three, [target; 0.5];
                                          weights=inverse_covariance_weights(cat(Σ, 1.0; dims=(1, 2)); nobs=nobs))
        nuisance = profile(fixed_objective, [(-1, 1), (-Inf, Inf), (0.5, 0.5)], [θ̂; 0.5], 1,
                           θ̂[1] .+ se .* [-1.0, 0.0, -0.0, 1.0]; config=config, workers=Int[])
        @test length(nuisance.values) == 3
        @test nuisance.delta ≈ [1.0, 0.0, 1.0] atol=1e-6
        @test all(x -> x[3] == 0.5, nuisance.x)
    end

    @testset "missed minima, unfinished searches, and disjoint sets" begin
        # Starting from a poor estimate, the profile finds lower losses.
        poor = θ̂ .+ [0.0, 0.2]
        p = profile(objective, bounds, poor, 1, grid[2:5]; config=config, workers=Int[])
        @test any(issue -> occursin("not the global minimum", issue), p.issues)
        @test_throws "reestimate" profile_interval(p)

        short = profile(objective, bounds, θ̂, 1, grid[3:4]; config=TikTakConfig(config; max_evals=3),
                        workers=Int[])
        @test all(==(:budget_exhausted), short.status)
        @test count(issue -> occursin("bounds the profile from above", issue), short.issues) == 2
        # An unfinished search only bounds the profile from above: inside if below the critical value.
        partial = profile_interval(short)
        @test partial.lower == (short.delta[1] <= 3.841458820694124 ? grid[3] : θ̂[1]) && !partial.lower_bounded

        # One parameter: each grid point is a single evaluation.
        two_valleys(x) = 4 * min(x[1]^2, (x[1] - 3)^2 + 0.5)
        valleys = profile(two_valleys, [(-5, 5)], [0.0], 1, -2:0.5:5; workers=Int[])
        @test valleys.delta ≈ two_valleys.(collect(-2:0.5:5))
        interval = profile_interval(valleys)
        # √ΔQ = 2|θ| is linear on each side, so the crossing is exact.
        @test [interval.lower, interval.upper] ≈ [-1, 1] .* sqrt(3.841458820694124) ./ 2
        @test interval.disjoint
        @test interval.lower_bounded && interval.upper_bounded
    end

    @testset "validation" begin
        options = (; config=config, workers=Int[])
        @test_throws "fixed" profile(objective, [(-1, 1), (θ̂[2], θ̂[2])], [θ̂[1], θ̂[2]], 2, [0.0]; options...)
        @test_throws "within the bounds" profile(objective, bounds, θ̂, 1, [2.0]; options...)
        @test_throws "within bounds" profile(objective, bounds, [2.0, 0.0], 1, [0.0]; options...)
        @test_throws "empty" profile(objective, bounds, θ̂, 1, Float64[]; options...)
        @test_throws "out of range" profile(objective, bounds, θ̂, 3, [0.0]; options...)
        @test_throws "by name" profile(objective, bounds, θ̂, "a", [0.0]; options...)
        @test_throws "no parameter is named" profile(objective, bounds, θ̂, "c", [0.0]; names=["a", "b"], options...)
        @test_throws "problem_id" profile(objective, bounds, θ̂, 1, [0.0]; run_dir=mktempdir(), options...)
        @test_throws "resume requires run_dir" profile(objective, bounds, θ̂, 1, [0.0]; resume=true, options...)
    end
end
