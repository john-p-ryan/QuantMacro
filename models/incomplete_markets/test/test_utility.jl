@testset "utility" begin
    c = [0.05, 0.5, 1.0, 3.0]

    @testset "CRRA with γ = $γ" for γ in [0.5, 1.0, 2.0, 5.0]
        crra = γ == 1 ? log.(c) : (c .^ (1 - γ) .- 1) ./ (1 - γ)
        for (_, m) in METHODS
            @test m.u.(c, γ) ≈ crra
        end
        # EGM also needs marginal utility and its inverse.
        h = 1e-6
        @test EGM.u_prime.(c, γ) ≈ (EGM.u.(c .+ h, γ) .- EGM.u.(c .- h, γ)) ./ 2h rtol=1e-6
        @test EGM.u_prime_inv.(EGM.u_prime.(c, γ), γ) ≈ c
    end

    @testset "continuous in γ at log utility" begin
        @test EGM.u(2.5, 1 + 1e-9) ≈ log(2.5) atol=1e-8
        @test EGM.u(2.5, 1 - 1e-9) ≈ log(2.5) atol=1e-8
    end

    @testset "linear extension below ε" begin
        # Continuous at ε with slope u'(ε), so infeasible consumption gets a finite but steep penalty.
        ε = 1e-7
        for γ in [1.0, 2.0], (_, m) in METHODS
            @test m.u(ε * (1 - 1e-12), γ) ≈ m.u(ε, γ)
            @test (m.u(ε / 2, γ) - m.u(-ε, γ)) / 1.5ε ≈ ε^(-γ)
            @test isfinite(m.u(-1.0, γ)) && m.u(-1.0, γ) < m.u(0.0, γ)
        end
        @test EGM.u_prime(0.0, 2.0) == EGM.u_prime(ε, 2.0)
    end

    @test_throws ErrorException EGM.u_prime_inv(0.0, 1.0)
end
