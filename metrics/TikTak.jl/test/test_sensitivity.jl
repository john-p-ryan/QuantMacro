@testset "moment-to-parameter sensitivity" begin
    J = [1.0 0.2; -0.5 2.0; 1.5 -1.0]
    θ = [0.4, 0.6]
    bounds = [(-Inf, Inf), (-Inf, Inf)]
    scales = [0.5, 2.0, 1.5]
    W = [2.0 0.4 -0.2; 0.4 1.5 0.3; -0.2 0.3 1.0]
    objective = MomentObjective(x -> J * x, J * θ; scales=scales, weights=W)
    jac = moment_jacobian(objective, θ, bounds; jacobian=J, names=["a", "b"])

    @testset "full weights and perturbed-target least squares" begin
        report = sensitivity_summary(jac, objective; moment_names=["mean", "variance", "covariance"],
                                     parameter_scales=[0.1, 2.0])
        raw_weights = Diagonal(1 ./ scales) * W * Diagonal(1 ./ scales)
        reference = (J' * raw_weights * J) \ (J' * raw_weights)
        @test report.target_sensitivity ≈ reference
        @test report.jacobian == J
        @test report.scaled_jacobian ≈ Diagonal(1 ./ scales) * J * Diagonal([0.1, 2.0])
        @test report.scaled_target_sensitivity ≈ Diagonal([10.0, 0.5]) * reference * Diagonal(scales)
        @test report.target_sensitivity * J ≈ Matrix(1.0I, 2, 2)
        @test report.rank == 2 && report.regular && isempty(report.issues)
        @test report.free == [1, 2] && report.on_bound == [false, false]
        @test report.parameter_names == ["a", "b"]
        @test report.moment_names == ["mean", "variance", "covariance"]
        @test report.parameter_scales == [0.1, 2.0] && report.moment_scales == scales

        # Independent re-fit via QR with a Cholesky weight factor, including
        # a nonzero residual. In a linear model the sensitivity is exact.
        L = cholesky(Symmetric(W)).U * Diagonal(1 ./ scales)
        target = J * θ + [0.1, -0.05, 0.2]
        baseline = qr(L * J) \ (L * target)
        for i in 1:3
            change = zeros(3)
            change[i] = 1e-4
            fitted = qr(L * J) \ (L * (target + change))
            @test fitted - baseline ≈ report.target_sensitivity * change atol=1e-12
        end
        # The same map propagates moment-error covariance in sandwich inference.
        Ω = [2.0 0.1 0.0; 0.1 1.0 0.2; 0.0 0.2 0.5]
        cov = parameter_covariance(jac, objective; covariance=Ω, nobs=1)
        @test report.target_sensitivity * Ω * report.target_sensitivity' ≈ cov.covariance
        display = sprint(show, MIME"text/plain"(), report)
        @test occursin("local linear approximation", display)
        @test occursin("covariance", display) && occursin("scaled sensitivity", display)
        @test occursin("rank 2 of 2", sprint(show, report))
    end

    @testset "weighting choices and original moment coordinates" begin
        model(x) = [x[1], 2x[1]]
        one_jac = moment_jacobian(model, [0.5], [(-Inf, Inf)]; jacobian=reshape([1.0, 2.0], 2, 1))
        for weights in ([1.0, 1.0], [3.0, 0.5], [1.0, 0.0])
            obj = MomentObjective(model, [0.5, 1.0]; weights=weights)
            s = sensitivity_summary(one_jac, obj)
            @test s.target_sensitivity ≈ [weights[1] 2weights[2]] / (weights[1] + 4weights[2])
        end
        # Full weights can give negative target influence even when all moments
        # increase with the parameter. Entries still refer to original moments.
        obj = MomentObjective(model, [0.5, 1.0]; weights=[1.0 -0.75; -0.75 1.0])
        @test sensitivity_summary(one_jac, obj).target_sensitivity ≈ [-0.25 0.625]
        # A singular weight matrix is valid if its weighted Jacobian has full rank.
        obj = MomentObjective(model, [0.5, 1.0]; weights=[1.0 1.0; 1.0 1.0])
        s = sensitivity_summary(one_jac, obj)
        @test s.regular && s.target_sensitivity ≈ [1/3 1/3]
    end

    @testset "normalization and unit invariance" begin
        raw = sensitivity_summary(jac, objective)
        @test raw.parameter_scales == ones(2) && raw.moment_names == ["m1", "m2", "m3"]
        for units in (:column, 3.0, [0.1, 100.0])
            s = sensitivity_summary(jac, objective; parameter_scales=units)
            @test s.target_sensitivity ≈ raw.target_sensitivity && s.rank == raw.rank
        end
        # Express moments in new units, changing their scales consistently.
        moment_units = [100.0, 0.01, 5.0]
        changed = MomentObjective(x -> moment_units .* (J * x), moment_units .* (J * θ);
                                  scales=moment_units .* scales, weights=W)
        changed_jac = moment_jacobian(changed, θ, bounds; jacobian=moment_units .* J)
        s = sensitivity_summary(changed_jac, changed)
        @test s.scaled_jacobian ≈ raw.scaled_jacobian
        @test s.scaled_target_sensitivity ≈ raw.scaled_target_sensitivity
        @test s.target_sensitivity .* moment_units' ≈ raw.target_sensitivity
        # Express parameters in new units and supply matching reporting units.
        units = [100.0, 0.01]
        new_jac = moment_jacobian(identity, θ .* units, bounds; jacobian=J ./ units')
        s = sensitivity_summary(new_jac, objective; parameter_scales=units)
        @test s.target_sensitivity ≈ units .* raw.target_sensitivity
        @test s.scaled_jacobian ≈ raw.scaled_jacobian
        @test s.scaled_target_sensitivity ≈ raw.scaled_target_sensitivity
        @test s.rank == raw.rank
        # Rescaling W alone does not change the linearized estimator.
        rescaled = MomentObjective(identity, objective.target; scales=scales, weights=7W)
        @test sensitivity_summary(jac, rescaled).target_sensitivity ≈ raw.target_sensitivity
    end

    @testset "AR(1) exact identification" begin
        ρ, σ = 0.65, 0.2
        v = σ^2 / (1 - ρ^2)
        model(x) = [x[2]^2 / (1 - x[1]^2), x[1] * x[2]^2 / (1 - x[1]^2)]
        ar_jac = moment_jacobian(model, [ρ, σ], [(0, 0.98), (0, Inf)])
        # Differentiate the inverse mapping: ρ = c/v, σ = √(v - c²/v).
        inverse_derivative = [-ρ/v 1/v; (1+ρ^2)/(2σ) -ρ/σ]
        for weights in ([1.0, 1.0], [3.0, 0.2], [2.0 0.5; 0.5 1.0])
            obj = MomentObjective(model, [v, ρ*v]; weights=weights, scales=[0.1, 2.0])
            @test sensitivity_summary(ar_jac, obj).target_sensitivity ≈ inverse_derivative rtol=1e-7
        end
    end

    @testset "fixed parameters and nonregular diagnostics" begin
        # Fixed parameters can occur between free parameters.
        fixed_jac = moment_jacobian(identity, [θ[1], 3.0, θ[2]],
                                    [bounds[1], (3, 3), bounds[2]];
                                    jacobian=hcat(J[:, 1], ones(3), J[:, 2]),
                                    names=["a", "fixed", "b"])
        fixed = sensitivity_summary(fixed_jac, objective; parameter_scales=[2.0, 10.0, 0.5])
        @test fixed.free == [1, 3] && fixed.parameter_names == ["a", "b"]
        @test fixed.parameter_scales == [2.0, 0.5]
        @test fixed.target_sensitivity ≈ sensitivity_summary(jac, objective).target_sensitivity

        for (matrix, weights, expected_rank) in (([1.0 1.0; 2.0 2.0], ones(2), 1),
                                                ([1.0 2.0], ones(1), 1),
                                                (J, zeros(3), 0))
            obj = MomentObjective(identity, zeros(size(matrix, 1)); weights=weights)
            deficient = moment_jacobian(identity, θ, bounds; jacobian=matrix)
            s = sensitivity_summary(deficient, obj)
            @test s.rank == expected_rank && !s.regular
            @test all(isnan, s.target_sensitivity) && all(isnan, s.scaled_target_sensitivity)
            @test s.jacobian == matrix && occursin("rank", only(s.issues))
        end
        near_singular = moment_jacobian(identity, θ, bounds; jacobian=[1.0 1.0; 0.0 1e-6])
        two = MomentObjective(identity, θ)
        @test sensitivity_summary(near_singular, two; rank_tol=1e-4).rank == 1
        @test sensitivity_summary(near_singular, two; rank_tol=1e-8).rank == 2
        for estimate in (0.0, 1.0)
            at_bound = moment_jacobian(identity, [estimate], [(0, 1)])
            s = sensitivity_summary(at_bound, MomentObjective(identity, [estimate]))
            @test s.on_bound == [true] && !s.regular && s.target_sensitivity ≈ ones(1, 1)
            @test occursin("unconstrained", only(s.issues))
        end
        staircase(x) = [round(x[1] / 1e-3) * 1e-3 + x[2], x[2]]
        unstable = moment_jacobian(staircase, [0.30049, 0.2], [(0, 1), (0, 1)])
        s = sensitivity_summary(unstable, two)
        @test !s.regular && any(issue -> occursin("changes by", issue), s.issues)
        @test occursin("not regular", sprint(show, MIME"text/plain"(), s))
    end

    @testset "validation and no model calls" begin
        never_call = MomentObjective(x -> error("must not evaluate model"), zeros(3); scales=scales, weights=W)
        @test sensitivity_summary(jac, never_call).regular
        @test_throws ArgumentError sensitivity_summary(jac, objective; moment_names=["only one"])
        for units in ([1.0], [1.0, 0.0], [-1.0, 1.0], Inf)
            @test_throws ArgumentError sensitivity_summary(jac, objective; parameter_scales=units)
        end
        for tol in (-1.0, 1.0, Inf, NaN)
            @test_throws ArgumentError sensitivity_summary(jac, objective; rank_tol=tol)
        end
        for tol in (-1.0, Inf, NaN)
            @test_throws ArgumentError sensitivity_summary(jac, objective; sensitivity_tol=tol)
        end
        @test_throws "different numbers of moments" sensitivity_summary(jac, MomentObjective(identity, [1.0]))
        isolated(x) = x == [0.5] ? [1.0] : throw(ModelEvaluationError("no equilibrium"))
        failed = moment_jacobian(isolated, [0.5], [(0, 1)])
        @test_throws "failed columns" sensitivity_summary(failed, MomentObjective(isolated, [1.0]))
    end
end
