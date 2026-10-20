@testset "inference" begin
    θ = [0.65, 0.2]
    bounds = [(0, 0.98), (0, Inf)]
    target = income_moments(θ)
    objective = MomentObjective(income_moments, target; scales=target)
    function income_jacobian(x)
        ρ, σ = x
        v = σ^2 / (1 - ρ^2)
        dv_dρ, dv_dσ = 2ρ * σ^2 / (1 - ρ^2)^2, 2σ / (1 - ρ^2)
        return [dv_dρ dv_dσ; v+ρ*dv_dρ ρ*dv_dσ; 2ρ*v+ρ^2*dv_dρ ρ^2*dv_dσ]
    end
    # Rank 2 of 3: only θ1 + θ2 enters the first two moments.
    deficient(x) = [x[1] + x[2], 2 * (x[1] + x[2]), x[3]]
    deficient_objective = MomentObjective(deficient, [1.0, 2.0, 0.5])
    deficient_jac = moment_jacobian(deficient_objective, [0.5, 0.5, 0.5], [(0, 1), (0, 1), (0, 1)])

    @testset "finite differences match the analytic AR(1) Jacobian" begin
        jac = moment_jacobian(objective, θ, bounds; names=["ρ", "σ"])
        @test jac.jacobian ≈ income_jacobian(θ) rtol=1e-8
        @test jac.schemes == [:central, :central] && isempty(jac.failures)
        @test all(<(1e-6), jac.sensitivity) && jac.names == ["ρ", "σ"]
        supplied = moment_jacobian(objective, θ, bounds; jacobian=income_jacobian)
        @test supplied.jacobian == income_jacobian(θ) && all(==(:supplied), supplied.schemes)
        fixed = moment_jacobian(x -> income_moments(x[1:2]), [θ; 3.0], [bounds; (3, 3)])
        @test fixed.free == [1, 2] && fixed.jacobian ≈ jac.jacobian
        @test moment_jacobian(objective, θ, bounds; mapper=pmap).jacobian == jac.jacobian
        @test occursin("central", sprint(show, MIME"text/plain"(), jac))
    end

    @testset "bounds, failed perturbations, and reproducibility" begin
        at_bound = moment_jacobian(objective, [0.98, 0.2], bounds)
        @test at_bound.schemes == [:backward, :central]
        @test at_bound.jacobian ≈ income_jacobian([0.98, 0.2]) rtol=1e-3
        narrow = moment_jacobian(x -> [x[1]^2], [0.5], [(0.5, 0.5 + 1e-7)])
        @test narrow.schemes == [:forward] && isapprox(narrow.jacobian[1], 1; rtol=1e-6)
        @test narrow.steps ≈ [1e-7] && isnan(narrow.sensitivity[1])

        kinked(x) = x[1] > 0.5 ? throw(ModelEvaluationError("no equilibrium")) : [x[1]^2, x[2]]
        jac = moment_jacobian(kinked, [0.5, 0.3], [(0, 1), (0, 1)]; check_factors=())
        @test jac.schemes == [:backward, :central] && isapprox(jac.jacobian[1, 1], 1; rtol=1e-4)
        @test length(jac.failures) == 1 && occursin("no equilibrium", only(jac.failures))
        isolated(x) = x == [0.5] ? [1.0] : throw(ModelEvaluationError("no equilibrium"))
        failed = moment_jacobian(isolated, [0.5], [(0, 1)])
        @test failed.schemes == [:failed] && all(isnan, failed.jacobian) && length(failed.failures) == 6
        @test_throws "failed columns" identification(failed, MomentObjective(isolated, [1.0]))

        @test_throws "evaluation failed" moment_jacobian(x -> throw(ModelEvaluationError()), [0.5], [(0, 1)])
        @test_throws "programming bug" moment_jacobian(x -> error("programming bug"), [0.5], [(0, 1)])
        @test_throws "not reproducible" moment_jacobian(x -> x .+ rand(), [0.5], [(0, 1)])
        calls = Ref(0)
        drifting(x) = [x[1] + ((calls[] += 1) > 3 ? 1e-6 : 0.0)]   # state advances after two calls
        @test_throws "not reproducible" moment_jacobian(drifting, [0.5], [(0, 1)])
        jittery = moment_jacobian(x -> [x[1], 1e-18 * rand()], [0.5], [(0, 1)])
        @test jittery.jacobian[1] ≈ 1
        @test_throws ArgumentError moment_jacobian(objective, [0.99, 0.2], bounds)
        @test_throws ArgumentError moment_jacobian(objective, θ, bounds; step=[1e-4, 0])
        @test_throws ArgumentError moment_jacobian(x -> x, [1.0], [(1, 1)])
    end

    @testset "step sensitivity flags discretized moments" begin
        # One grid point of the staircase lies between the base and the doubled step.
        staircase(x) = [round(x[1] / 1e-3) * 1e-3 + x[2], x[2]]
        jac = moment_jacobian(staircase, [0.30049, 0.2], [(0, 1), (0, 1)])
        @test jac.sensitivity[1] >= 0.5 && jac.sensitivity[2] < 1e-8
        coarse = moment_jacobian(staircase, [0.30049, 0.2], [(0, 1), (0, 1)]; step=0.05)
        @test isapprox(coarse.jacobian[1, 1], 1; rtol=0.05) && coarse.sensitivity[1] < 0.05
    end

    @testset "identification under explicit normalizations" begin
        jac = moment_jacobian(objective, θ, bounds; jacobian=income_jacobian)
        criterion = income_jacobian(θ) ./ target
        report = identification(jac, objective)
        @test report.singular_values ≈ svdvals(criterion ./ norm.(eachcol(criterion))')
        @test report.rank == 2 && report.condition ≈ report.singular_values[1] / report.singular_values[2]
        @test identification(jac, objective; parameter_scales=:raw).singular_values ≈ svdvals(criterion)
        custom = identification(jac, objective; parameter_scales=[0.1, 0.01])
        @test custom.singular_values ≈ svdvals(criterion .* [0.1, 0.01]')
        @test custom.parameter_scales == [0.1, 0.01]
        # The default is invariant to parameter units: measure σ in percent.
        percent = moment_jacobian(x -> income_moments([x[1], x[2] / 100]), [0.65, 20.0], [(0, 0.98), (0, Inf)])
        @test identification(percent, objective).singular_values ≈ report.singular_values rtol=1e-6

        weak = identification(deficient_jac, deficient_objective)
        @test weak.rank == 2 && weak.singular_values[3] < 1e-8 && weak.condition > 1e8
        @test abs(dot(weak.directions[:, 3], [1, -1, 0])) ≈ sqrt(2) atol=1e-6
        @test occursin("rank 2 of 3", sprint(show, MIME"text/plain"(), weak))
        sum_objective = MomentObjective(x -> [sum(x)], [0.5])
        wide = identification(moment_jacobian(sum_objective, [0.2, 0.3], [(0, 1), (0, 1)]), sum_objective)
        @test wide.rank == 1 && wide.singular_values[2] == 0 && wide.condition == Inf
        @test_throws ArgumentError identification(jac, objective; parameter_scales=[1.0, -1.0])
        @test_throws "different numbers of moments" identification(jac, sum_objective)
    end

    @testset "sandwich covariance" begin
        # Sample mean: the standard error is σ/√N.
        mean_objective = MomentObjective(identity, [1.5])
        mean_jac = moment_jacobian(mean_objective, [1.5], [(-Inf, Inf)])
        mean = parameter_covariance(mean_jac, mean_objective; covariance=fill(4.0, 1, 1), nobs=100)
        @test mean.standard_errors ≈ [0.2] && mean.regular && isempty(mean.issues)
        @test wald_intervals(mean) ≈ [1.5-1.96*0.2 1.5+1.96*0.2] rtol=1e-4

        # Independent reference in raw moment units with explicit inverses.
        J = income_jacobian(θ)
        jac = moment_jacobian(objective, θ, bounds; jacobian=income_jacobian)
        sigma = [2.0 0.5 0.1; 0.5 1.0 0.3; 0.1 0.3 0.8] .* (target * target')
        omega = sigma / 500
        scales, W = [0.1, 0.05, 0.02], [1.0 0.2 0.0; 0.2 2.0 0.1; 0.0 0.1 0.5]
        D⁻¹ = Diagonal(1 ./ scales)
        raw_weights = D⁻¹ * W * D⁻¹
        bread = inv(J' * raw_weights * J)
        reference = bread * J' * raw_weights * omega * raw_weights * J * bread
        weighted = MomentObjective(income_moments, target; scales=scales, weights=W)
        result = parameter_covariance(jac, weighted; covariance=sigma, nobs=500)
        @test result.covariance ≈ reference rtol=1e-10
        @test result.standard_errors ≈ sqrt.(diag(reference)) && result.regular
        diagonal = MomentObjective(income_moments, target; scales=scales, weights=diag(W))
        diagonal_weights = D⁻¹ * Diagonal(diag(W)) * D⁻¹
        diagonal_bread = inv(J' * diagonal_weights * J)
        @test parameter_covariance(jac, diagonal; covariance=sigma, nobs=500).covariance ≈
              diagonal_bread * J' * diagonal_weights * omega * diagonal_weights * J * diagonal_bread rtol=1e-10
        # Efficient weights W = S⁻¹ give (J'Ω⁻¹J)⁻¹.
        efficient = MomentObjective(income_moments, target; scales=scales, weights=inv(D⁻¹ * omega * D⁻¹))
        @test parameter_covariance(jac, efficient; covariance=sigma, nobs=500).covariance ≈
              inv(J' * inv(omega) * J) rtol=1e-8
        # Exactly identified: weights and scales are irrelevant, V = J⁻¹ΩJ⁻ᵀ.
        J2, omega2 = J[1:2, :], omega[1:2, 1:2]
        for (weights, moment_scales) in ((nothing, nothing), ([3.0, 0.2], [0.5, 2.0]), ([2.0 0.5; 0.5 1.0], nothing))
            two = MomentObjective(x -> income_moments(x)[1:2], target[1:2]; weights=weights, scales=moment_scales)
            two_jac = moment_jacobian(two, θ, bounds; jacobian=J2)
            @test parameter_covariance(two_jac, two; covariance=sigma[1:2, 1:2], nobs=500).covariance ≈
                  inv(J2) * omega2 * inv(J2)' rtol=1e-8
        end
    end

    @testset "rank deficiency and bounds are flagged" begin
        result = parameter_covariance(deficient_jac, deficient_objective; covariance=Matrix(1.0I, 3, 3), nobs=10)
        @test result.rank == 2 && !result.regular && all(isnan, result.standard_errors)
        @test_throws "rank 2" wald_intervals(result)

        lower_objective = MomentObjective(identity, [0.0])
        for (estimate, flagged) in ((0.0, true), (0.3, true), (1.0, false))
            jac = moment_jacobian(lower_objective, [estimate], [(0, Inf)])
            result = parameter_covariance(jac, lower_objective; covariance=fill(4.0, 1, 1), nobs=100)
            @test result.standard_errors ≈ [0.2] && result.near_bound == [flagged] && result.regular == !flagged
        end
        # Roundoff in the column of a parameter without effect is normalized to unit norm,
        # so only the step sensitivity reveals it.
        inert(x) = [x[1] + 0.1sin(x[1]), 2x[1], x[1]^2 + sqrt(x[2])^2 - x[2]]
        inert_objective = MomentObjective(inert, inert([0.5, 0.7]))
        inert_jac = moment_jacobian(inert_objective, [0.5, 0.7], [(-Inf, Inf), (0, Inf)])
        inert_result = parameter_covariance(inert_jac, inert_objective; covariance=Matrix(1.0I, 3, 3), nobs=100)
        @test inert_jac.sensitivity[2] > 0.1 && !inert_result.regular
        @test any(issue -> occursin("changes by", issue), inert_result.issues)
        @test identification(inert_jac, inert_objective).parameter_scales[2] > 1e6

        on_bound = parameter_covariance(moment_jacobian(lower_objective, [0.0], [(0, Inf)]), lower_objective;
                                        covariance=fill(4.0, 1, 1), nobs=100)
        @test on_bound.issues == ["θ1 is on a bound"]
        @test_throws "on a bound" wald_intervals(on_bound)
        @test occursin("not regular", sprint(show, MIME"text/plain"(), on_bound))

        jac = moment_jacobian(objective, θ, bounds; jacobian=income_jacobian)
        for (covariance, nobs) in ((Matrix(1.0I, 2, 2), 10), ([1.0 0 0; 1 1 0; 0 0 1], 10),
                                   (-Matrix(1.0I, 3, 3), 10), (Matrix(1.0I, 3, 3), 0))
            @test_throws ArgumentError parameter_covariance(jac, objective; covariance=covariance, nobs=nobs)
        end
    end
end
