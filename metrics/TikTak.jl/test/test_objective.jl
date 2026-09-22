@testset "MomentObjective" begin
    @testset "weighted criterion and validation" begin
        target = [1.0, 2.0]
        weights = [2 0.5; 0.5 1]
        objective = MomentObjective(identity, target; weights=weights, scales=[2, 3])
        x = [2.0, 4.0]
        errors = (x .- target) ./ [2, 3]
        ev = objective(x)
        @test ev isa Evaluation
        @test ev.value ≈ dot(errors, weights * errors)
        @test ev.moments == x
        @test dot(ev.residuals, ev.residuals) ≈ ev.value
        for bad in ([-1, 1], [1 2; 2 1], [1 2; 0 1], [1, 2, 3], [NaN, 1])
            @test_throws ArgumentError MomentObjective(identity, target; weights=bad)
        end
        @test_throws ArgumentError MomentObjective(x -> [1], target)(x)
        @test_throws ArgumentError MomentObjective(x -> 1.0, target)(x)
        @test_throws ArgumentError MomentObjective(identity, Float64[])
        @test_throws ArgumentError MomentObjective(identity, [1.0, Inf])
        @test_throws ArgumentError MomentObjective(identity, target; scales=[0, 1])
        @test_throws ModelEvaluationError MomentObjective(x -> [NaN, 1.0], target)(x)
        @test MomentObjective(identity, target)(target).value == 0
        diagonal = MomentObjective(identity, target; weights=[4, 9])
        @test diagonal(x).value ≈ 4 * 1 + 9 * 4
        spec = TikTak.specification(objective)
        @test spec["weights"] == [[2.0, 0.5], [0.5, 1.0]] && spec["scales"] == [2.0, 3.0]
        @test TikTak.specification(diagonal)["weights"] == [4.0, 9.0]
    end

    @testset "Evaluation" begin
        e = Evaluation(1; moments=[1, 2])
        @test e.value === 1.0 && e.moments == [1.0, 2.0] && e.residuals === nothing
        @test_throws ArgumentError Evaluation(1; moments=[1 2; 3 4])
        @test sprint(showerror, ModelEvaluationError("no equilibrium")) == "no equilibrium"
    end
end
