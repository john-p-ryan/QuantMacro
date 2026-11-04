@testset "inverse-covariance weights" begin
    Σ = [4.0 1.2 0.4; 1.2 1.0 0.3; 0.4 0.3 0.25]
    nobs = 50
    Ω = Σ / nobs
    R = Ω ./ sqrt.(diag(Ω) * diag(Ω)')
    scales = [2.0, 0.1, 5.0]
    errors = [0.3, -0.1, 0.2]

    @testset "efficient weights are converted to scale units" begin
        w = inverse_covariance_weights(Σ; nobs=nobs, moment_names=["mean", "var", "acf"])
        @test w.precision ≈ inv(Ω) && w.shrinkage == 0 && !w.diagonal
        @test w.standard_deviations ≈ sqrt.(diag(Ω)) && w.names == ["mean", "var", "acf"]
        @test w.eigenvalues ≈ eigvals(Symmetric(R)) && w.condition ≈ cond(R) && w.rank == 3
        @test w.regularized_condition ≈ w.condition
        @test R * w.directions ≈ w.directions .* w.eigenvalues'
        objective = MomentObjective(identity, zeros(3); scales=scales, weights=w)
        @test objective.weights ≈ Diagonal(scales) * inv(Ω) * Diagonal(scales)
        @test objective.weights == objective.weights'
        # The criterion is (m - target)'Ω⁻¹(m - target) whatever the scales.
        @test objective(errors).value ≈ errors' * inv(Ω) * errors
        @test MomentObjective(identity, zeros(3); weights=w)(errors).value ≈ objective(errors).value
        @test_throws "3 moments but the target has 2" MomentObjective(identity, zeros(2); weights=w)
    end

    @testset "diagonal weights and shrinkage towards the diagonal" begin
        d = inverse_covariance_weights(Σ; nobs=nobs, diagonal=true)
        @test d.precision ≈ 1 ./ diag(Ω) && d.diagonal && d.shrinkage == 1
        @test d.condition ≈ cond(R)    # diagnostics still describe the correlations
        objective = MomentObjective(identity, zeros(3); scales=scales, weights=d)
        @test objective.weights isa Vector && objective.weights ≈ scales .^ 2 ./ diag(Ω)
        for λ in (0.25, 1.0)
            s = inverse_covariance_weights(Σ; nobs=nobs, shrinkage=λ)
            @test s.precision ≈ inv((1 - λ) * Ω + λ * Diagonal(diag(Ω)))
            @test s.shrinkage == λ && s.regularized_condition ≈ cond((1 - λ) * R + λ * I)
        end
    end

    @testset "condition-number target" begin
        κ = cond(R) / 2
        w = inverse_covariance_weights(Σ; nobs=nobs, max_condition=κ)
        @test 0 < w.shrinkage < 1
        @test cond((1 - w.shrinkage) * R + w.shrinkage * I) ≈ κ rtol=1e-10
        @test w.regularized_condition ≈ κ rtol=1e-10
        # The smallest sufficient shrinkage: any less violates the target.
        λ = 0.99 * w.shrinkage
        @test cond((1 - λ) * R + λ * I) > κ
        @test inverse_covariance_weights(Σ; nobs=nobs, max_condition=2 * cond(R)).shrinkage == 0
        @test inverse_covariance_weights(Σ; nobs=nobs, max_condition=κ, shrinkage=0.9).shrinkage == 0.9
        @test inverse_covariance_weights(Σ; nobs=nobs, max_condition=1).precision ≈ Diagonal(1 ./ diag(Ω))
    end

    @testset "regularization does not depend on moment units" begin
        units = [100.0, 0.01, 3.0]
        a = inverse_covariance_weights(Σ; nobs=nobs, max_condition=2.0)
        b = inverse_covariance_weights(units .* Σ .* units'; nobs=nobs, max_condition=2.0)
        @test b.shrinkage ≈ a.shrinkage && b.eigenvalues ≈ a.eigenvalues && b.directions ≈ a.directions
        @test b.precision ≈ a.precision ./ (units .* units')
    end

    @testset "singular and invalid covariances" begin
        # The third moment is the sum of the first two.
        collinear = [1.0 0.0; 0.0 1.0; 1.0 1.0] * [2.0 0.3; 0.3 1.0] * [1.0 0.0 1.0; 0.0 1.0 1.0]
        @test_throws "numerically singular" inverse_covariance_weights(collinear; nobs=1)
        @test_throws "+0.7385 m3 -0.5505 m1 -0.3892 m2" inverse_covariance_weights(collinear; nobs=1)
        @test_throws "after shrinkage" inverse_covariance_weights(collinear; nobs=1, shrinkage=1e-12)
        w = inverse_covariance_weights(collinear; nobs=1, max_condition=1e3)
        @test w.rank == 2 && w.condition > 1e10
        @test w.regularized_condition ≈ 1e3 rtol=1e-6
        null = sqrt.(diag(collinear)) .* [1.0, 1.0, -1.0]
        @test abs(dot(w.directions[:, 1], null)) ≈ norm(null)
        @test inverse_covariance_weights(collinear; nobs=1, diagonal=true).precision ≈ 1 ./ diag(collinear)

        @test_throws "inverse covariance: m2" inverse_covariance_weights([1.0 0 0; 0 0 0; 0 0 1]; nobs=1)
        for (covariance, options) in (([1.0 2.0; 2.0 1.0], (;)), ([1.0 0.5; 0.0 1.0], (;)),
                                      (zeros(0, 0), (;)), (ones(2, 3), (;)), ([1.0 0; 0 1], (; shrinkage=1.5)),
                                      ([1.0 0; 0 1], (; max_condition=0.5)), ([1.0 0; 0 1], (; rank_tol=1.0)))
            @test_throws ArgumentError inverse_covariance_weights(covariance; nobs=1, options...)
        end
        @test_throws ArgumentError inverse_covariance_weights(Σ; nobs=0)
        # Indefinite at a small scale, which an absolute tolerance on Σ would accept.
        @test_throws "positive semidefinite" inverse_covariance_weights(1e-12 .* [1.0 2.0; 2.0 1.0]; nobs=1)
    end

    @testset "sandwich inference with inverse-covariance weights" begin
        J = [1.0 0.5; 0.2 1.0; 0.7 -0.3]
        θ = [0.1, 0.2]
        bounds = [(-Inf, Inf), (-Inf, Inf)]
        for (options, precision) in (((;), inv(Ω)), ((; shrinkage=0.5), inv(0.5 * Ω + 0.5 * Diagonal(diag(Ω)))))
            w = inverse_covariance_weights(Σ; nobs=nobs, options...)
            objective = MomentObjective(x -> J * x, J * θ; scales=scales, weights=w)
            jac = moment_jacobian(objective, θ, bounds; jacobian=J)
            bread = inv(J' * precision * J)
            reference = bread * J' * precision * Ω * precision * J * bread
            @test parameter_covariance(jac, objective; covariance=Σ, nobs=nobs).covariance ≈ reference rtol=1e-10
        end
        efficient = MomentObjective(x -> J * x, J * θ; weights=inverse_covariance_weights(Σ; nobs=nobs))
        jac = moment_jacobian(efficient, θ, bounds; jacobian=J)
        @test parameter_covariance(jac, efficient; covariance=Σ, nobs=nobs).covariance ≈ inv(J' * inv(Ω) * J)
    end

    @testset "copying an objective with new weights" begin
        base = MomentObjective(income_moments, [1.0, 2.0, 3.0]; scales=scales, weights=[1.0, 2.0, 3.0])
        w = inverse_covariance_weights(Σ; nobs=nobs)
        copied = MomentObjective(base; weights=w)
        @test copied.model === base.model && copied.target == base.target && copied.scales == base.scales
        @test copied.weights ≈ MomentObjective(income_moments, base.target; scales=scales, weights=w).weights
        @test MomentObjective(base).weights == base.weights
        rescaled = MomentObjective(base; scales=1.0)
        @test rescaled.scales == ones(3) && rescaled.weights == base.weights
    end

    @testset "display" begin
        w = inverse_covariance_weights(Σ; nobs=nobs, moment_names=["mean", "var", "acf"])
        text = sprint(show, MIME"text/plain"(), w)
        @test occursin("full inverse covariance", text) && occursin("rank 3 of 3", text)
        @test occursin("-0.7071 mean", text)
        shrunk = sprint(show, MIME"text/plain"(), inverse_covariance_weights(Σ; nobs=nobs, max_condition=2.0))
        @test occursin("towards the diagonal", shrunk) && occursin("2.0 after shrinkage", shrunk)
        @test occursin("inverse variances", sprint(show, MIME"text/plain"(),
                                                   inverse_covariance_weights(Σ; nobs=nobs, diagonal=true)))
        @test sprint(show, w) == "InverseCovarianceWeights(3 moments, shrinkage 0.0)"
    end

    @testset "simulation noise adds to the moment covariance" begin
        w = inverse_covariance_weights(Σ; nobs=nobs, n_simulations=4)
        @test w.precision ≈ inv(1.25 * Ω)
        Σₛ = Matrix(Diagonal(diag(Σ)))
        explicit = inverse_covariance_weights(Σ; nobs=nobs, simulation_covariance=Σₛ, simulation_nobs=200)
        @test explicit.precision ≈ inv(Ω + Σₛ / 200) && explicit.standard_deviations ≈ sqrt.(diag(Ω + Σₛ / 200))
        @test_throws "passed together" inverse_covariance_weights(Σ; nobs=nobs, simulation_covariance=Σₛ)
    end
end
