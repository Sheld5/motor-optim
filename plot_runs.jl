using BOSS
using JLD2
using Plots
using Distributions

include("./Surrogate_Q_volne_parametry.jl")


function compare_methods(init_data)
    dir = "./motor-optim/experiments/data-withparam"
    fitness = BOSS.LinFitness([0., -1.])
    y_max = ModelParam.y_max()

    kernels = [BOSS.Matern12Kernel, BOSS.Matern32Kernel, BOSS.Matern52Kernel]
    kernames = [split(string(ker), '.')[end] for ker in kernels]

    files = readdir(dir)
    rand = [load(dir*"/"*f) for f in files if startswith(f, "rand_")]
    par = [load(dir*"/"*f) for f in files if startswith(f, "par_")]
    parbi = [load(dir*"/"*f) for f in files if startswith(f, "parbi_")]
    mle = [[load(dir*"/"*f) for f in files if startswith(f, "mle_"*ker*'_')] for ker on kernames]
    bi = [[load(dir*"/"*f) for f in files if startswith(f, "bi_"*ker*'_')] for ker in kernames]
    gp = [[load(dir*"/"*f) for f in files if startswith(f, "gp_"*ker*'_')] for ker in kernames]
    gpbi = [[load(dir*"/"*f) for f in files if startswith(f, "gpbi_"*ker*'_')] for ker in kernames]

    rand_bsf = bsf_series.(rand, Ref(fitness), Ref(y_max), Ref(init_data))
    par_bsf = bsf_series.(par, Ref(fitness), Ref(y_max), Ref(init_data))
    parbi_bsf = bsf_series.(parbi, Ref(fitness), Ref(y_max), Ref(init_data))
    mle_bsf = [bsf_series.(mle[i], Ref(fitness), Ref(y_max), Ref(init_data)) for i in eachindex(kernames)]
    bi_bsf = [bsf_series.(bi[i], Ref(fitness), Ref(y_max), Ref(init_data)) for i in eachindex(kernames)]
    gp_bsf = [bsf_series.(gp[i], Ref(fitness), Ref(y_max), Ref(init_data)) for i in eachindex(kernames)]
    gpbi_bsf = [bsf_series.(gpbi[i], Ref(fitness), Ref(y_max), Ref(init_data)) for i in eachindex(kernames)]

    # TODO
    rand_bsf, mle_bsf, bi_bsf, gp_bsf, gpbi_bsf =
        skip_inconsistent_runs([rand_bsf, par_bsf, parbi_bsf, mle_bsf..., bi_bsf..., gp_bsf..., gpbi_bsf...])

    p = plot(; title="fitness", ylabel="Tav", xlabel="iteration")
    plot_runs!(p, deepcopy(rand_bsf); label="RAND")
    plot_runs!(p, deepcopy(par_bsf); label="PAR (MLE)")
    plot_runs!(p, deepcopy(parbi_bsf); label="PAR (BI)")
    for i in eachindex(kernames) plot_runs!(p, deepcopy(mle_bsf[i]); label="SEMI (MLE)") end
    for i in eachindex(kernames) plot_runs!(p, deepcopy(bi_bsf[i]); label="SEMI (BI)") end
    for i in eachindex(kernames) plot_runs!(p, deepcopy(gp_bsf[i]); label="GP (MLE)") end
    for i in eachindex(kernames) plot_runs!(p, deepcopy(gpbi_bsf[i]); label="GP (BI)") end
    display(p)

    # p1 = plot(; title="(median,min,max) fitness", ylabel="Tav", xlabel="iteration")
    # plot_runs!(p1, deepcopy(rand_bsf); label="RAND")
    # display(p1)
    # p2 = plot(; title="(median,min,max) fitness", ylabel="Tav", xlabel="iteration")
    # plot_runs!(p2, deepcopy(mle_bsf); label="MLE")
    # display(p2)
    # p3 = plot(; title="(median,min,max) fitness", ylabel="Tav", xlabel="iteration")
    # plot_runs!(p3, deepcopy(bi_bsf); label="BI")
    # display(p3)
    # p4 = plot(; title="(median,min,max) fitness", ylabel="Tav", xlabel="iteration")
    # plot_runs!(p4, deepcopy(gp_bsf); label="GP (MLE)")
    # display(p4)
    # p5 = plot(; title="(median,min,max) fitness", ylabel="Tav", xlabel="iteration")
    # plot_runs!(p5, deepcopy(gpbi_bsf); label="GP (BI)")
    # display(p5)

    nothing
end

function plot_runs!(p, runs; label=nothing)
    iterations = first(runs)[1]
    
    # assert all runs have the same number of iterations
    # @assert all(r -> r[1]==iterations, runs)
    inconsistent = get_inconsistent_runs(runs, length(iterations))
    if (length(inconsistent) > 0)
        @warn "Inconsistent number of iteration among runs. $(length(inconsistent)) runs will be skipped!
        Inconsistent run indices: $(inconsistent)."
    end
    runs = [runs[i] for i in eachindex(runs) if !(i in inconsistent)]

    bsf = [r[2] for r in runs]
    INFEASIBLE = 100.
    for r in eachindex(bsf)
        for i in eachindex(bsf[r])
            if isnothing(bsf[r][i])
                bsf[r][i] = INFEASIBLE
            else
                bsf[r][i] *= -1.
            end
        end
    end

    # mins = minimum.(((r[i] for r in bsf) for i in 1:length(iterations)))
    # maxs = maximum.(((r[i] for r in bsf) for i in 1:length(iterations)))
    mins = (a -> quantile(a, 0.1)).(((r[i] for r in bsf) for i in 1:length(iterations)))
    maxs = (a -> quantile(a, 0.9)).(((r[i] for r in bsf) for i in 1:length(iterations)))
    meds = median.(((r[i] for r in bsf) for i in 1:length(iterations)))

    # @show last(meds)

    # m = maximum(maxs)
    plot!(p, iterations, meds;
        ribbons=(meds.-mins, maxs.-meds),
        label,
        ylimits=(0.,INFEASIBLE),
        markerstrokecolor=:auto,
        # yaxis=:log,
    )
end

function bsf_series(res, fitness, y_max, init_data)
    X, Y = res["X"], res["Y"]
    @assert size(X)[2] == size(Y)[2]
    feasible(y) = all(y .<= y_max)

    iters = size(X)[2] - init_data
    iteration = [i for i in 0:iters]

    opt_x = nothing
    fit = [fitness(y) for y in eachcol(Y[:,1:init_data]) if feasible(y)]
    bsf = Union{Nothing, Float64}[isempty(fit) ? nothing : maximum(fit)]
    for i in 1:iters
        y = Y[:,init_data+i]
        if feasible(y) && (isnothing(last(bsf)) || (fitness(y) > last(bsf)))
            push!(bsf, fitness(y))
            opt_x = X[:,init_data+i]
        else
            push!(bsf, last(bsf))
        end
    end

    x, y = result(X, Y)
    if (y[2] < -3.)
        @show x,y
    end
    return [iteration, bsf]
end

function get_inconsistent_runs(runs, iters)
    return [i for i in eachindex(runs) if length(runs[i][1]) != iters]
end

function skip_inconsistent_runs(bsf_series)
    @show [length(bsf) for bsf in bsf_series]
    @assert all(bsf -> length(bsf) == length(first(bsf_series)), bsf_series)
    iters = length(bsf_series[1][1][1])
    inconsistent = Set(reduce(vcat, [get_inconsistent_runs(bsf, iters) for bsf in bsf_series]))
    @info "Inconsistent indices skipped: $(inconsistent)."
    consistent = [i for i in eachindex(first(bsf_series)) if !(i in inconsistent)]
    bsf_series = [bsf[consistent] for bsf in bsf_series]
    return bsf_series
end

function get_opt()
    dir = "data03"
    f = "mle_10.jld2"

    data = load(dir*"/"*f)
    result(data)
end

result(data) = result(data["X"], data["Y"])

function result(X, Y)
    # @warn "Using hard-coded fitness."
    fit = BOSS.LinFitness([0., -1.])

    @assert size(X)[2] == size(Y)[2]
    isempty(X) && return nothing

    feasible = BOSS.is_feasible.(eachcol(Y), Ref(ModelParam.y_max()))
    fitness = fit.(eachcol(Y))
    fitness[.!feasible] .= -Inf
    best = argmax(fitness)

    feasible[best] || return nothing
    return X[:,best], Y[:,best]
end
