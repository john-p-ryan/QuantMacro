@testset "two-step estimation" begin
    # Linear moments m(θ) = Xθ: the first step is least squares weighted by the
    # objective's scales and the second step is GLS, both in closed form.
    X = [1.0 0.0; 0.0 1.0; 1.0 1.0; 1.0 -0.5]
    target = X * [0.3, -0.2] .+ [0.02, -0.01, 0.03, -0.02]
    Σ = [1.0 0.5 0.2 0.1; 0.5 2.0 0.3 0.0; 0.2 0.3 1.5 0.4; 0.1 0.0 0.4 0.8]
    nobs = 200
    Ω = Σ / nobs
    scales = [0.5, 1.0, 2.0, 1.0]
    bounds = [(-1, 1), (-1, 1)]
    config = TikTakConfig(n_samples=16, n_local=3, local_max_evals=400, x_tol=1e-8, f_tol=1e-14)
    calls = Ref(0)
    linear(x) = (calls[] += 1; X * x)
    objective = MomentObjective(linear, target; scales=scales)
    gls(precision) = (X' * precision * X) \ (X' * precision * target)
    second_dirs(dir) = filter(startswith("step2-"), readdir(dir))
    saved_points(dir) = [Vector{Float64}(p) for p in get(history(dir), "points")]

    @testset "closed-form estimates, J statistic, and standard errors" begin
        dir = mktempdir()
        result = two_step(objective, bounds; covariance=Σ, nobs=nobs, config=config, run_dir=dir,
                          problem_id="linear-v1", workers=Int[])
        @test result.status == :completed && has_solution(result)
        @test result.first.x ≈ gls(Diagonal(1 ./ scales .^ 2)) atol=1e-6
        @test result.x ≈ gls(inv(Ω)) atol=1e-6
        @test result.x == result.second.x
        residual = target - X * gls(inv(Ω))
        @test result.j_statistic ≈ residual' * inv(Ω) * residual rtol=1e-8
        @test result.j_statistic == result.fun && result.degrees_of_freedom == 2
        @test result.covariance == Σ && result.nobs == nobs && result.weights.shrinkage == 0
        @test result.objective.model === linear && result.objective.scales == scales
        jac = moment_jacobian(result.objective, result.x, bounds; jacobian=X)
        estimates = parameter_covariance(jac, result.objective; covariance=result.covariance, nobs=result.nobs)
        @test estimates.covariance ≈ inv(X' * inv(Ω) * X)

        # Separate, versioned runs; the second starts from the first-step estimate.
        @test isdir(joinpath(dir, "step1")) && length(second_dirs(dir)) == 1
        second_dir = joinpath(dir, only(second_dirs(dir)))
        @test result.second.run_dir == second_dir && result.first.run_dir == joinpath(dir, "step1")
        @test to_unit(BoxTransform(bounds), result.first.x) in saved_points(second_dir)
        @test get(history(second_dir), "specification")["problem_id"] == "linear-v1/step2"
        # Every second-step loss was computed under the second-step criterion.
        @test all(r -> r.value ≈ result.objective(r.parameters).value, load_evaluations(second_dir; status=:ok))
        text = sprint(show, MIME"text/plain"(), result)
        @test occursin("J statistic", text) && occursin("2 degrees of freedom", text)
    end

    @testset "resume reuses both runs; new weights start a new second step" begin
        dir = mktempdir()
        options = (; nobs=nobs, config=config, run_dir=dir, problem_id="linear-resume", workers=Int[])
        result = two_step(objective, bounds; covariance=Σ, options...)
        before = calls[]
        again = two_step(objective, bounds; covariance=Σ, resume=true, options...)
        @test calls[] == before && again.x == result.x && again.second.n_evals == result.second.n_evals
        @test length(second_dirs(dir)) == 1
        n_saved = length(load_evaluations(result.second.run_dir))

        diagonal = Matrix(Diagonal(diag(Σ)))
        changed = two_step(objective, bounds; covariance=diagonal, resume=true, options...)
        @test length(second_dirs(dir)) == 2 && changed.second.run_dir != result.second.run_dir
        @test changed.first.x == result.first.x
        @test changed.x ≈ gls(inv(diagonal / nobs)) atol=1e-6
        @test all(r -> r.value ≈ changed.objective(r.parameters).value,
                  load_evaluations(changed.second.run_dir; status=:ok))
        @test length(load_evaluations(result.second.run_dir)) == n_saved
        @test_throws "already exists" two_step(objective, bounds; covariance=Σ, options...)
    end

    @testset "a covariance function is evaluated at the first-step estimate" begin
        seen = Vector{Float64}[]
        result = two_step(objective, bounds; covariance=θ -> (push!(seen, copy(θ)); Σ), nobs=nobs,
                          config=config, n_warm=1, workers=Int[])
        @test seen == [result.first.x]
        @test result.x ≈ gls(inv(Ω)) atol=1e-6
        # Only the first-step estimate joins the Sobol points.
        @test length(saved_points(result.second.run_dir)) == config.n_samples + 1
        @test_throws "real matrix" two_step(objective, bounds; covariance=θ -> 1.0, nobs=nobs, config=config,
                                            workers=Int[])
    end

    @testset "simulation noise in the weights and standard errors" begin
        R = 4
        result = two_step(objective, bounds; covariance=Σ, nobs=nobs, n_simulations=R, config=config, workers=Int[])
        total = (1 + 1 / R) * Ω
        # A scalar multiple of Ω leaves the GLS estimate unchanged but rescales J.
        @test result.x ≈ gls(inv(Ω)) atol=1e-6
        residual = target - X * gls(inv(Ω))
        @test result.j_statistic ≈ residual' * inv(total) * residual rtol=1e-8
        @test result.simulation_covariance == Σ && result.simulation_nobs == R * nobs
        jac = moment_jacobian(result.objective, result.x, bounds; jacobian=X)
        estimates = parameter_covariance(jac, result)
        @test estimates.covariance ≈ inv(X' * inv(total) * X)
        # With efficient weights the simulation part is V / (R + 1).
        @test estimates.simulation_covariance ≈ estimates.covariance / (R + 1)
        @test result.n_simulations == R
        # A covariance reevaluated at the estimate carries the stored simulation term along.
        doubled = parameter_covariance(jac, result; covariance=2Σ)
        @test doubled.covariance ≈ 2 * estimates.covariance && doubled.simulation_covariance ≈ 2 * estimates.simulation_covariance
        # An explicit simulation term replaces the stored one: Ω = 2Ω₀ under weights (1.25Ω₀)⁻¹.
        @test parameter_covariance(jac, result; n_simulations=1).covariance ≈ 1.6 * estimates.covariance
        @test_throws "second-step estimate" parameter_covariance(
            moment_jacobian(result.objective, [0.0, 0.0], bounds; jacobian=X), result)

        seen = Vector{Float64}[]
        Σₛ = Matrix(Diagonal(diag(Σ)))
        explicit = two_step(objective, bounds; covariance=Σ, nobs=nobs, config=config, n_warm=1, workers=Int[],
                            simulation_covariance=θ -> (push!(seen, copy(θ)); Σₛ), simulation_nobs=1000)
        @test seen == [explicit.first.x] && explicit.simulation_covariance == Σₛ && explicit.simulation_nobs == 1000
        @test explicit.x ≈ gls(inv(Ω + Σₛ / 1000)) atol=1e-6
        plain = two_step(objective, bounds; covariance=Σ, nobs=nobs, config=config, workers=Int[])
        @test plain.simulation_covariance === nothing && plain.simulation_nobs === nothing
        @test parameter_covariance(moment_jacobian(plain.objective, plain.x, bounds; jacobian=X), plain).covariance ≈
              inv(X' * inv(Ω) * X)
        partial = two_step(objective, bounds; covariance=Σ, nobs=nobs, config=TikTakConfig(config; max_evals=5),
                           workers=Int[])
        @test_throws "has not run" parameter_covariance(jac, partial)
        @test_throws "passed together" two_step(objective, bounds; covariance=Σ, nobs=nobs, simulation_nobs=10,
                                                workers=Int[])
    end

    @testset "the second step waits for a complete first step" begin
        dir = mktempdir()
        partial = two_step(objective, bounds; covariance=Σ, nobs=nobs, config=TikTakConfig(config; max_evals=10),
                           run_dir=dir, problem_id="linear-partial", workers=Int[])
        @test partial.status == :budget_exhausted && partial.second === nothing && !has_solution(partial)
        @test startswith(partial.message, "first step") && isempty(second_dirs(dir)) && isnan(partial.j_statistic)
        @test occursin("first step", sprint(show, MIME"text/plain"(), partial))
        finished = two_step(objective, bounds; covariance=Σ, nobs=nobs, config=config, run_dir=dir,
                            problem_id="linear-partial", resume=true, workers=Int[])
        @test finished.status == :completed
        @test finished.x ≈ gls(inv(Ω)) atol=1e-6
    end

    @testset "regularized weights and reweighted screening" begin
        coarse = MomentObjective(x -> X * x .+ 0.001, target)
        result = two_step(objective, bounds; covariance=Σ, nobs=nobs, weight_options=(; shrinkage=0.5),
                          screen_objective=coarse, config=config, workers=Int[])
        precision = inv(0.5 * Ω + 0.5 * Diagonal(diag(Ω)))
        @test result.weights.shrinkage == 0.5 && isnan(result.j_statistic)
        @test result.x ≈ gls(precision) atol=1e-6
        saved = get(history(result.second.run_dir), "specification")["screen_moments"]
        @test reduce(hcat, [Float64.(row) for row in saved["weights"]]) ≈ precision   # unit scales
        @test occursin("shrinkage 0.5", sprint(show, MIME"text/plain"(), result))
    end

    @testset "validation before any search" begin
        dir = mktempdir()
        singular = [1.0 0 0; 0 1 0; 0 0 1; 1 1 0] * Σ[1:3, 1:3] * [1.0 0 0 1; 0 1 0 1; 0 0 1 0]
        for covariance in (Σ[1:3, 1:3], singular)
            @test_throws ArgumentError two_step(objective, bounds; covariance=covariance, nobs=nobs,
                                                run_dir=dir, problem_id="checked", workers=Int[])
        end
        @test isempty(readdir(dir))
        regularized = two_step(objective, bounds; covariance=singular, nobs=nobs, config=config,
                               weight_options=(; max_condition=1e4), workers=Int[])
        @test regularized.status == :completed
        @test regularized.weights.regularized_condition ≈ 1e4 rtol=1e-6
        @test_throws "MomentObjective" two_step(objective, bounds; covariance=Σ, nobs=nobs,
                                                screen_objective=x -> 0.0)
        @test_throws "n_warm" two_step(objective, bounds; covariance=Σ, nobs=nobs, n_warm=0)
        @test_throws "problem_id" two_step(objective, bounds; covariance=Σ, nobs=nobs, run_dir=dir)
        @test_throws "resume requires run_dir" two_step(objective, bounds; covariance=Σ, nobs=nobs, resume=true)
    end

    @testset "distributed two-step run" begin
        θ = [0.6, 0.3]
        income = MomentObjective(income_moments, income_moments(θ); scales=income_moments(θ))
        covariance = [2.0 0.5 0.1; 0.5 1.0 0.3; 0.1 0.3 0.8] .* (income.target * income.target')
        result = two_step(income, [(0, 0.95), (0.05, 1)]; covariance=covariance, nobs=500, workers=workers(),
                          config=TikTakConfig(n_samples=16, n_local=2, x_tol=1e-7, f_tol=1e-12))
        @test result.status == :completed && result.j_statistic < 1e-6
        @test result.x ≈ θ atol=1e-4
    end
end
