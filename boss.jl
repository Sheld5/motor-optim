using BOSS
using Distributions
using OptimizationOptimJL
# using NLopt
# using OptimizationMOI, Juniper, Ipopt
using PRIMA
using JLD2

include("motor_problem.jl")

function get_priors(x_dim, y_dim)
    # θ: A1, A2, A3
    param_range = ModelParam.param_range()
    θ_priors = [truncated(Normal(mid, dif/3); lower=0.) for (mid, dif) in zip(mean(param_range), param_range[2]-param_range[1])]
    
    # x: nk, dk, Ds, Q
    domain = ModelParam.domain()
    length_scale_priors = fill(Product([truncated(Normal(0., dif/3); lower=0.) for dif in (domain[2][i]-domain[1][i] for i in 1:x_dim)]), y_dim)
    
    # y: dP, Tav
    noise_var_priors = fill(Dirac(1e-8), y_dim)

    return θ_priors, length_scale_priors, noise_var_priors
end

function get_problem(X, Y)
    x_dim, y_dim = size(X)[1], size(Y)[1]

    objective(x) = ModelParam.calc(x...)
    parametric(x, θ) = ModelParam.calc(x..., θ...)
    constraints(x) = [ModelParam.check_feas(x...)...]

    domain = BOSS.Domain(;
        bounds = ModelParam.domain(),
        discrete = ModelParam.discrete_dims(),
        cons = constraints,
    )
    
    θ_priors, length_scale_priors, noise_var_priors = get_priors(x_dim, y_dim)

    model = BOSS.Semiparametric(
        BOSS.NonlinModel(;
            predict=parametric,
            param_priors=θ_priors,
        ),
        BOSS.Nonparametric(;
            length_scale_priors,
        ),
    )

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

function test_script(problem=nothing; iters=1, mle=true, random=false)
    if isnothing(problem)
        X, Y = get_data()
        problem = get_problem(X, Y)
    end

    model_fitter = nothing
    if mle
        model_fitter = BOSS.OptimizationMLE(;
            algorithm=LBFGS(),
            multistart=200,
            parallel=true,
            apply_softplus=true,
            softplus_params=fill(true, 3),
            x_tol=1e-3,
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
            multistart=200,
            parallel=true,
            rhoend=1e-3,
        )
    end

    # acq_maximizer = BOSS.GridAM(;
    #     problem,
    #     steps=[1., 0.01, 0.01, 0.01],
    #     parallel=true,
    # )
    # @show length(acq_maximizer.points)

    # acq_maximizer = BOSS.NLoptAM(;
    #     algorithm=:LN_COBYLA,
    #     multistart=32,
    #     parallel=true,
    #     xtol_abs=1e-3,
    #     # maxtime=2.,
    # )

    # nl_solver = OptimizationMOI.MOI.OptimizerWithAttributes(
    #     Ipopt.Optimizer,
    #     "print_level" => 0,
    # )
    # nl_solver = OptimizationMOI.MOI.OptimizerWithAttributes(
    #     NLopt.Optimizer,
    #     "algorithm" => :GN_ORIG_DIRECT,
    #     "xtol_abs" => 1e-3,
    # )
    # minlp_solver = OptimizationMOI.MOI.OptimizerWithAttributes(
    #     Juniper.Optimizer,
    #     "nl_solver" => nl_solver,
    #     "log_levels" => Symbol[],
    #     # "time_limit" => 20.,
    #     "atol" => 1e-18,
    # )
    # acq_maximizer = BOSS.OptimizationAM(;
    #     algorithm=minlp_solver,
    #     multistart=1,
    #     parallel=false,
    #     autodiff=AutoForwardDiff(),
    # )

    term_cond = BOSS.IterLimit(iters)

    options = BOSS.BossOptions(;
        info=true,
        debug=false,
        ϵ_samples=1,  # only affects MLE
    )

    boss!(problem; model_fitter, acq_maximizer, term_cond, options)
    @show result(problem)
    return problem
end

function result(problem)
    feasible = BOSS.is_feasible.(eachcol(problem.data.Y), Ref(problem.y_max))
    fitness = problem.fitness.(eachcol(problem.data.Y))
    fitness[.!feasible] .= -Inf
    best = argmax(fitness)
    return problem.data.X[:,best], problem.data.Y[:,best]
end



# - - - RUN - - -

function data_dict(data::BOSS.ExperimentDataPost)
    return Dict(
        "X"=>data.X,
        "Y"=>data.Y,
        "θ"=>data.θ,
        "length_scales"=>data.length_scales,
        "noise_vars"=>data.noise_vars,
    )
end

function runopt()
    runs = 20
    iters = 20

    for r in 1:runs
        res = test_script(; iters, mle=true, random=true)
        save("./data/rand_$r.jld2", data_dict(res.data))
        
        res = test_script(; iters, mle=true)
        save("./data/mle_$r.jld2", data_dict(res.data))

        res = test_script(; iters, mle=false)
        save("./data/bi_$r.jld2", data_dict(res.data))
    end
end
