using BOSS
using Distributions
using PRIMA
using JLD2

include("motor_problem.jl")
include("data.jl")

function get_priors(x_dim, y_dim)
    # θ: A1, A2, A3
    param_range = ModelParam.param_range()
    θ_priors = [truncated(Normal(mid, dif/3); lower=0.) for (mid, dif) in zip(mean(param_range), param_range[2]-param_range[1])]
    
    # x: nk, dk, Ds, Q
    domain = ModelParam.domain()
    length_scale_priors = fill(Product([truncated(Normal(0., dif/3); lower=0.) for dif in (domain[2][i]-domain[1][i] for i in 1:x_dim)]), y_dim)
    
    # y: dP, Tav
    noise_var_priors = fill(Dirac(1e-8), y_dim)
    # noise_var_priors = [truncated(Normal(0., 1.); bound=0.), truncated(Normal(0., 0.01); bound=0.)]

    return θ_priors, length_scale_priors, noise_var_priors
end

function get_problem(X, Y; param=true)
    x_dim, y_dim = size(X)[1], size(Y)[1]

    objective(x) = ModelParam.calc(x...)
    parametric(x, θ) = ModelParam.calc(x..., θ...)
    
    domain = get_domain()
    
    θ_priors, length_scale_priors, noise_var_priors = get_priors(x_dim, y_dim)

    model = nothing
    if param
        model = BOSS.Semiparametric(
            BOSS.NonlinModel(;
                predict=parametric,
                param_priors=θ_priors,
            ),
            BOSS.Nonparametric(;
                length_scale_priors,
            ),
        )
    else
        model = BOSS.Nonparametric(;
            length_scale_priors,
        )
    end

    BOSS.OptimizationProblem(;
        fitness = BOSS.LinFitness([0., -1.]),
        f = objective,
        domain,
        y_max = ModelParam.y_max(),
        model,
        noise_var_priors,
        data = BOSS.ExperimentDataPrior(X, Y),
    )
end

"""
Run BOSS on the motor problem.

Use keyword `param` to change between the Semiparametric model and sole GP.
Use keyword `mle` to change between MLE and BI.
Use keyword `random` to change between maximizing the acquisition function and random sampling.
"""
function test_script(problem=nothing; init_data=1, iters=1, mle=true, random=false, param=true)
    if isnothing(problem)
        X, Y = get_data(init_data, get_domain())
        problem = get_problem(X, Y; param)
    end
    @assert param == !(problem.model isa BOSS.Nonparametric)

    model_fitter = nothing
    if mle
        model_fitter = BOSS.NewuoaMLE(PRIMA;
            multistart=200,
            parallel=true,
            apply_softplus=true,
            softplus_params=(param ? fill(true, 3) : nothing),
            rhoend=1e-3,
        )
    else
        model_fitter = BOSS.TuringBI(;
            sampler=BOSS.PG(20),
            warmup=400,
            samples_in_chain=10,
            chain_count=8,
            leap_size=5,
            parallel=true,
        )
    end

    acq_maximizer = nothing
    if random
        acq_maximizer = BOSS.RandomSelectAM()
    else
        acq_maximizer = BOSS.CobylaAM(PRIMA;
            multistart=200, # Make sure `multistart` >> 60 as Cobyla is not optimizing over the discrete `nk`.
            parallel=true,
            rhoend=1e-3,
        )
    end

    acquisition = BOSS.ExpectedImprovement(;
        cons_safe=true,
        ϵ_samples=1,
    )

    term_cond = BOSS.IterLimit(iters)

    options = BOSS.BossOptions(;
        info=true,
        debug=false,
    )

    boss!(problem; model_fitter, acq_maximizer, acquisition, term_cond, options)
    @show result(problem)
    return problem
end

"""
Return the best found point.
"""
function result(problem)
    feasible = BOSS.is_feasible.(eachcol(problem.data.Y), Ref(problem.y_max))
    fitness = problem.fitness.(eachcol(problem.data.Y))
    fitness[.!feasible] .= -Inf
    best = argmax(fitness)
    return problem.data.X[:,best], problem.data.Y[:,best]
end



# - - - RUN - - -

"""
Script to run multiple BOSS runs.
"""
function runopt()
    runs = 10
    iters = 10

    for r in 1:runs
        # new random data
        X, Y = get_data(2, get_domain())

        problem = get_problem(deepcopy.((X, Y))...)
        res = test_script(problem; iters, mle=true, random=true)
        save("./data/rand_$r.jld2", data_dict(res.data))
        
        problem = get_problem(deepcopy.((X, Y))...)
        res = test_script(problem; iters, mle=true)
        save("./data/mle_$r.jld2", data_dict(res.data))

        # problem = get_problem(deepcopy.((X, Y))...)
        # res = test_script(problem; iters, mle=false)
        # save("./data/bi_$r.jld2", data_dict(res.data))

        # problem = get_problem(deepcopy.((X, Y))...; param=false)
        # res = test_script(problem; iters, mle=true, param=false)
        # save("./data/gp_$r.jld2", data_dict(res.data))
    end
end

# function contexp()
#     dir = "./data03"
#     runs = 20
#     iters = 10
#     init_data = 4
    
#     files = readdir(dir)
#     mle = [load(dir*"/"*f) for f in files if startswith(f, "mle")]
#     @assert length(mle) == runs
#     @assert size(first(mle)["X"])[2] == init_data + iters
#     starts = [(r["X"][:,1:init_data], r["Y"][:,1:init_data]) for r in mle]
#     @assert length(starts) == runs

#     for r in 1:runs
#         X, Y = starts[r]

#         problem = get_problem(deepcopy.((X, Y))...; param=false)
#         res = test_script(problem; iters, mle=true, param=false)
#         save(dir*"/gp_$r.jld2", data_dict(res.data))
#     end
# end
