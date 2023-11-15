
function compare_methods(init_data)
    dir = "./data"
    fitness = BOSS.LinFitness([0., -1.])
    y_max = ModelParam.y_max()

    files = readdir(dir)
    rand = [load(dir*"/"*f) for f in files if startswith(f, "rand")]
    mle = [load(dir*"/"*f) for f in files if startswith(f, "mle")]
    bi = [load(dir*"/"*f) for f in files if startswith(f, "bi")]
    gp = [load(dir*"/"*f) for f in files if startswith(f, "gp")]

    rand_bsf = bsf_series.(rand, Ref(fitness), Ref(y_max), Ref(init_data))
    mle_bsf = bsf_series.(mle, Ref(fitness), Ref(y_max), Ref(init_data))
    bi_bsf = bsf_series.(bi, Ref(fitness), Ref(y_max), Ref(init_data))
    gp_bsf = bsf_series.(gp, Ref(fitness), Ref(y_max), Ref(init_data))

    plot(; title="(median,min,max) fitness", ylabel="Tav", xlabel="iteration")
    plot_runs!(rand_bsf; label="RAND")
    plot_runs!(mle_bsf; label="MLE")
    plot_runs!(bi_bsf; label="BI")
    plot_runs!(gp_bsf; label="GP (MLE)")
end

function plot_runs!(runs; label=nothing)
    # assert all runs have the same number of iterations
    iterations = first(runs)[1]
    @assert all(r -> r[1]==iterations, runs)
    bsf = [r[2] for r in runs]

    UNFEASIBLE = 1000.
    for r in eachindex(bsf)
        for i in eachindex(bsf[r])
            if isnothing(bsf[r][i])
                bsf[r][i] = UNFEASIBLE
            else
                bsf[r][i] *= -1.
            end
        end
    end

    mins = minimum.(((r[i] for r in bsf) for i in 1:length(iterations)))
    maxs = maximum.(((r[i] for r in bsf) for i in 1:length(iterations)))
    meds = median.(((r[i] for r in bsf) for i in 1:length(iterations)))

    @show last(meds)

    # m = maximum(maxs)
    plot!(iterations, meds;
        yerror=(meds.-mins, maxs.-meds),
        label,
        ylimits=(0.,1000.),
        markerstrokecolor=:auto,
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

    println("$opt_x $(last(bsf))")
    return [iteration, bsf]
end

function get_opt()
    dir = "data03"
    f = "mle_10.jld2"

    data = load(dir*"/"*f)
    result(data)
end

function result(data)
    fit = BOSS.LinFitness([0., -1.])

    X, Y = data["X"], data["Y"]
    @assert size(X)[2] == size(Y)[2]
    isempty(X) && return nothing

    feasible = BOSS.is_feasible.(eachcol(Y), Ref(ModelParam.y_max()))
    fitness = fit.(eachcol(Y))
    fitness[.!feasible] .= -Inf
    best = argmax(fitness)

    feasible[best] || return nothing
    return X[:,best], Y[:,best]
end
