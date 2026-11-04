@testset "simulation noise and the delta method" begin
    θ = [0.65, 0.2]
    bounds = [(0, 0.98), (0, Inf)]
    target = income_moments(θ)
    function income_jacobian(x)
        ρ, σ = x
        v = σ^2 / (1 - ρ^2)
        dv_dρ, dv_dσ = 2ρ * σ^2 / (1 - ρ^2)^2, 2σ / (1 - ρ^2)
        return [dv_dρ dv_dσ; v+ρ*dv_dρ ρ*dv_dσ; 2ρ*v+ρ^2*dv_dρ ρ^2*dv_dσ]
    end
    J = income_jacobian(θ)
    sigma = [2.0 0.5 0.1; 0.5 1.0 0.3; 0.1 0.3 0.8] .* (target * target')
    sigma_sim = [1.0 0.2 0.0; 0.2 1.5 0.1; 0.0 0.1 0.6] .* (target * target')
    scales, W = [0.1, 0.05, 0.02], [1.0 0.2 0.0; 0.2 2.0 0.1; 0.0 0.1 0.5]
    objective = MomentObjective(income_moments, target; scales=scales, weights=W)
    jac = moment_jacobian(objective, θ, bounds; jacobian=income_jacobian, names=["ρ", "σ"])
    # Independent reference in raw moment units with explicit inverses.
    raw_weights = Diagonal(1 ./ scales) * W * Diagonal(1 ./ scales)
    bread = inv(J' * raw_weights * J)
    sandwich(omega) = bread * J' * raw_weights * omega * raw_weights * J * bread

    @testset "sampling and simulation parts of the sandwich" begin
        result = parameter_covariance(jac, objective; covariance=sigma, nobs=500,
                                      simulation_covariance=sigma_sim, simulation_nobs=2000)
        @test result.sampling_covariance ≈ sandwich(sigma / 500) rtol=1e-10
        @test result.simulation_covariance ≈ sandwich(sigma_sim / 2000) rtol=1e-10
        @test result.covariance ≈ sandwich(sigma / 500 + sigma_sim / 2000) rtol=1e-10
        @test result.covariance ≈ result.sampling_covariance + result.simulation_covariance
        combined = parameter_covariance(jac, objective; covariance=sigma / 500 + sigma_sim / 2000, nobs=1)
        @test combined.covariance ≈ result.covariance && combined.simulation_covariance === nothing
        @test result.parameters == θ && result.free == [1, 2] && result.lower == [0, 0]
        text = sprint(show, MIME"text/plain"(), result)
        @test occursin("simulation share", text)
        @test occursin("not supplied separately", sprint(show, MIME"text/plain"(), combined))

        # R simulated samples like the data: Ω = (1 + 1/R) Σ / nobs.
        mean_objective = MomentObjective(identity, [1.5])
        mean_jac = moment_jacobian(mean_objective, [1.5], [(-Inf, Inf)])
        mean = parameter_covariance(mean_jac, mean_objective; covariance=fill(4.0, 1, 1), nobs=100, n_simulations=4)
        @test mean.covariance ≈ fill(0.05, 1, 1) && mean.simulation_covariance ≈ fill(0.01, 1, 1)
        @test mean.sampling_covariance ≈ fill(0.04, 1, 1)

        options = (; covariance=sigma, nobs=500)
        @test_throws "not both" parameter_covariance(jac, objective; options..., n_simulations=2,
                                                     simulation_covariance=sigma, simulation_nobs=10)
        @test_throws "passed together" parameter_covariance(jac, objective; options..., simulation_covariance=sigma)
        @test_throws "passed together" parameter_covariance(jac, objective; options..., simulation_nobs=10)
        @test_throws "simulation_nobs must be positive" parameter_covariance(jac, objective; options...,
                                                                             simulation_covariance=sigma,
                                                                             simulation_nobs=0)
        @test_throws "simulation_covariance must be symmetric" parameter_covariance(
            jac, objective; options..., simulation_covariance=[1.0 0 0; 1 1 0; 0 0 1], simulation_nobs=10)
        @test_throws "n_simulations" parameter_covariance(jac, objective; options..., n_simulations=0)
        @test_throws "n_simulations" parameter_covariance(jac, objective; options..., n_simulations=true)
    end

    estimates = parameter_covariance(jac, objective; covariance=sigma, nobs=500,
                                     simulation_covariance=sigma_sim, simulation_nobs=2000)

    @testset "delta method against analytic gradients" begin
        # Unconditional variance and half-life of the AR(1).
        g(x) = [x[2]^2 / (1 - x[1]^2), log(0.5) / log(x[1])]
        gradient(x) = [2x[1]*x[2]^2/(1-x[1]^2)^2 2x[2]/(1-x[1]^2);
                       -log(0.5) / (x[1] * log(x[1])^2) 0.0]
        G = gradient(θ)
        derived = delta_method(estimates, g; names=["variance", "half-life"])
        @test derived.estimate ≈ g(θ) && derived.names == ["variance", "half-life"]
        @test derived.jacobian ≈ G rtol=1e-7
        @test derived.covariance ≈ G * estimates.covariance * G' rtol=1e-6
        @test derived.sampling_covariance ≈ G * estimates.sampling_covariance * G' rtol=1e-6
        @test derived.simulation_covariance ≈ G * estimates.simulation_covariance * G' rtol=1e-6
        @test derived.standard_errors ≈ sqrt.(diag(G * estimates.covariance * G')) rtol=1e-6
        @test derived.regular && isempty(derived.issues) && all(<(1e-5), derived.sensitivity)
        @test wald_intervals(derived) ≈ [derived.estimate .- 1.959963984540054 .* derived.standard_errors derived.estimate .+ 1.959963984540054 .* derived.standard_errors]
        @test occursin("half-life", sprint(show, MIME"text/plain"(), derived))

        supplied = delta_method(estimates, g; jacobian=gradient)
        @test supplied.jacobian == G && supplied.covariance ≈ G * estimates.covariance * G'
        @test supplied.names == ["g1", "g2"]
        # Linear functions are exact; a scalar g may supply its gradient as a vector.
        A = [1.0 -2.0]
        linear = delta_method(estimates, x -> 1.0 * x[1] - 2.0 * x[2]; jacobian=vec(A))
        @test linear.covariance ≈ A * estimates.covariance * A' && length(linear.estimate) == 1
        @test delta_method(estimates, x -> x[1] - 2x[2]).covariance ≈ A * estimates.covariance * A' rtol=1e-7
        @test delta_method(estimates, x -> x[1] - 2x[2]; mapper=pmap).covariance ≈ linear.covariance rtol=1e-7
        @test_throws "one row per derived quantity" delta_method(estimates, g; jacobian=ones(3, 2))
        @test derived.parameter_names == ["ρ", "σ"] && derived.free == [1, 2]
        @test delta_method(estimates, x -> [x[1] x[2]]).jacobian ≈ I(2) rtol=1e-8
        @test_throws "real number or a real array" delta_method(estimates, x -> (x[1], x[2]))
    end

    @testset "fixed parameters, bounds, and failures" begin
        fixed_objective = MomentObjective(x -> income_moments(x[1:2]), target; scales=scales, weights=W)
        fixed_jac = moment_jacobian(fixed_objective, [θ; 3.0], [bounds; (3, 3)]; names=["ρ", "σ", "κ"])
        fixed = parameter_covariance(fixed_jac, fixed_objective; covariance=sigma, nobs=500)
        # g sees the full vector; the fixed κ has no uncertainty.
        derived = delta_method(fixed, x -> x[3] * x[1] + x[2])
        @test derived.jacobian ≈ [3.0 1.0] rtol=1e-8
        @test derived.covariance ≈ [3.0 1.0] * fixed.covariance * [3.0, 1.0] rtol=1e-7
        @test derived.simulation_covariance === nothing

        # At the bound ρ = 0.98 the derivative is one-sided and the estimate is flagged.
        edge_jac = moment_jacobian(objective, [0.98, 0.2], bounds; jacobian=income_jacobian, names=["ρ", "σ"])
        edge = parameter_covariance(edge_jac, objective; covariance=sigma, nobs=500)
        edged = delta_method(edge, x -> x[1] + x[2])
        @test !edge.regular && !edged.regular && edged.issues[1:length(edge.issues)] == edge.issues
        @test_throws "not valid" wald_intervals(edged)

        # Rank deficiency propagates as NaN.
        deficient(x) = [x[1] + x[2], 2 * (x[1] + x[2]), x[3]]
        deficient_objective = MomentObjective(deficient, [1.0, 2.0, 0.5])
        deficient_jac = moment_jacobian(deficient_objective, [0.5, 0.5, 0.5], [(0, 1), (0, 1), (0, 1)])
        rank_deficient = parameter_covariance(deficient_jac, deficient_objective; covariance=Matrix(1.0I, 3, 3),
                                              nobs=10)
        nan = delta_method(rank_deficient, x -> x[1] - x[2])
        @test all(isnan, nan.standard_errors) && !nan.regular && any(issue -> occursin("rank 2", issue), nan.issues)

        kinked(x) = x[1] > θ[1] ? throw(ModelEvaluationError("no equilibrium")) : x[1]^2
        one_sided = delta_method(estimates, kinked)
        @test one_sided.jacobian[1] ≈ 2θ[1] rtol=1e-4
        @test one_sided.regular
        isolated(x) = x[1] == θ[1] ? 1.0 : throw(ModelEvaluationError("no equilibrium"))
        failed = delta_method(estimates, isolated; names=["q"])
        @test !failed.regular && any(issue -> occursin("ρ failed", issue), failed.issues)
        # Only the doubled step reaches the next stair.
        staircase(x) = round((x[1] - 0.64965) / 1e-3) * 1e-3
        unstable = delta_method(estimates, staircase; step=[1e-4, 1e-4])
        @test !unstable.regular && any(issue -> occursin("changes by", issue), unstable.issues)
        @test_throws "could not be evaluated" delta_method(estimates, x -> throw(ModelEvaluationError("no")))
        @test_throws "programming bug" delta_method(estimates, x -> error("programming bug"))
        @test_throws "not reproducible" delta_method(estimates, x -> x[1] + rand())
    end
end
