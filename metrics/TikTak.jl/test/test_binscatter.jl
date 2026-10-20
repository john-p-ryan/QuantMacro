@testset "bin-scatter panels" begin
    saved(x, m; status=:ok, screening=false) = SavedEvaluation(
        "test", 1, screening, Float64[], Float64.(x), status, 0.0,
        m === nothing ? nothing : Float64.(m), nothing, nothing, 0.0)
    rows = [saved([x, 5 - x, 7], [2x, x^2]) for x in 1:4]
    data = binscatter_data(rows; nbins=2, parameter_names=["a", "b", "fixed"],
                          moment_names=["linear", "square"])
    @test size(data.panels) == (2, 3)
    @test data.parameter_names == ["a", "b", "fixed"]
    @test data.moment_names == ["linear", "square"]
    @test data.parameters == [1, 2, 3] && data.n_evaluations == 4
    @test data.panels[1, 1] == (x=[1.5, 3.5], y=[3.0, 7.0], count=[2, 2])
    @test data.panels[2, 1].y == [2.5, 12.5]
    @test data.panels[1, 2] == (x=[1.5, 3.5], y=[7.0, 3.0], count=[2, 2])
    @test data.panels[2, 3] == (x=[7.0], y=[7.5], count=[4])
    @test rows[1].parameters == [1, 4, 7] && rows[1].moments == [2, 1]
    reordered = binscatter_data(rows; nbins=2, parameters=[3, 1])
    @test reordered.parameters == [3, 1]
    @test reordered.panels[:, 2] == data.panels[:, 1]
    @test length(binscatter_data(rows; nbins=1).panels[1, 1].x) == 1
    @test length(binscatter_data(rows; nbins=100).panels[1, 1].x) == 4

    tied = [saved([x], [i]) for (i, x) in enumerate([0, 0, 0, 1, 2, 2])]
    @test binscatter_data(tied; nbins=4).panels[1, 1].count == [3, 3]
    bins = binscatter_data(tied; nbins=6).panels[1, 1]
    @test bins.x == [0, 1, 2] && bins.count == [3, 1, 2]
    @test bins.y ≈ [2, 4, 5.5]
    @test sum(bins.y .* bins.count) / sum(bins.count) ≈ 3.5

    mixed = [rows; saved([0, 0, 0], [999]; screening=true);
             saved([0, 0, 0], [999, 999]; status=:failed);
             saved([0, 0, 0], nothing); saved([0, 0, 0], [NaN, 0]);
             saved([Inf, 0, 0], [0, 0])]
    @test binscatter_data(mixed; nbins=2).panels == data.panels
    @test binscatter_data(mixed; stage=:screening).panels[1, 1].y == [999]
    @test_throws ArgumentError binscatter_data(SavedEvaluation[])
    @test_throws ArgumentError binscatter_data([saved([1], nothing)])
    @test_throws ArgumentError binscatter_data(rows; stage=:all)
    @test_throws ArgumentError binscatter_data(rows; nbins=0)
    @test_throws ArgumentError binscatter_data(rows; nbins=-1)
    @test_throws ArgumentError binscatter_data(rows; parameter_names=["a"])
    @test_throws ArgumentError binscatter_data(rows; moment_names=["m"])
    for indices in (Int[], [0], [4], [1, 1], [1.5])
        @test_throws ArgumentError binscatter_data(rows; parameters=indices)
    end
    @test_throws ArgumentError binscatter_data([rows; saved([1, 2], [1, 2])])
    @test_throws ArgumentError binscatter_data([rows; saved([1, 2, 3], [1])])

    @testset "saved runs and retained snapshots" begin
        calls = Ref(0)
        objective = MomentObjective([0.5, 1.0]) do x
            calls[] += 1
            [x[1], x[1]^2]
        end
        for retain in (false, true)
            result = minimize(objective, [(0, 1)]; workers=Int[], store_evaluations=retain,
                              config=TikTakConfig(n_samples=8, n_local=1, max_evals=20))
            before = calls[]
            from_result = binscatter_data(result; nbins=3)
            from_disk = binscatter_data(result.run_dir; nbins=3)
            @test from_result == from_disk
            @test from_result.n_evaluations == result.n_evals
            @test calls[] == before
        end
    end

    @test TikTak.BINSCATTER_BACKEND[] === nothing
    @test_throws ArgumentError binscatter(rows)
    # Load Plots only after checking the optional-dependency error path.
    @eval using Plots
    @test Base.get_extension(TikTak, :TikTakPlotsExt) !== nothing
    figure = binscatter(rows; nbins=2, parameters=[1, 2],
                        parameter_names=["a", "b", "fixed"],
                        moment_names=["linear", "square"], size=(700, 500),
                        color=:red, markersize=3)
    @test length(figure.subplots) == 4
    @test figure[:size] == (700, 500)
    for i in 1:2, j in 1:2
        subplot = figure.subplots[(i - 1) * 2 + j]
        @test subplot[:xaxis][:guide] == data.parameter_names[j]
        @test subplot[:yaxis][:guide] == data.moment_names[i]
        series = only(subplot.series_list)
        @test series[:x] == data.panels[i, j].x
        @test series[:y] == data.panels[i, j].y
        @test series[:seriestype] == :scatter
        @test series[:markersize] == 3
    end
    mktempdir() do dir
        output = joinpath(dir, "binscatter.svg")
        savefig(figure, output)
        @test filesize(output) > 0
    end
    @test length(binscatter([rows[1]]).subplots) == 6
end
