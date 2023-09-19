using BOSS
using Distributions
using OptimizationOptimJL
using NLopt

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
    constraints(x) = ModelParam.check_feas(x...)

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

function test_script(problem=nothing; iters=1)
    if isnothing(problem)
        X, Y = get_data()
        problem = get_problem(X, Y)
    end

    model_fitter = BOSS.OptimizationMLE(;
        algorithm=LBFGS(),
        multistart=200,
        parallel=true,
        apply_softplus=true,
        softplus_params=fill(true, 3),
        x_tol=1e-3,
    )
    # model_fitter = BOSS.TuringBI(;
    #     sampler=BOSS.PG(20),
    #     warmup=400,
    #     samples_in_chain=10,
    #     chain_count=8,
    #     leap_size=5,
    #     parallel=true,
    # )

    # acq_maximizer = BOSS.GridAM(;
    #     problem,
    #     steps=[1., 0.01, 0.01, 0.01],
    #     parallel=false,
    # )
    acq_maximizer = BOSS.NLoptAM(;
        algorithm=:LN_COBYLA,
        multistart=200,
        parallel=true,
        xtol_abs=1e-3,
        maxeval=100,
    )
    # maxtime = 2.
    # local_opt = NLopt.Opt(:LN_BOBYQA, BOSS.x_dim(problem))
    # local_opt.xtol_abs = 1e-3
    # local_opt.maxtime = maxtime
    # acq_maximizer = BOSS.NLoptAM(;
    #     algorithm=:AUGLAG,
    #     local_optimizer=local_opt,
    #     maxtime,
    #     multistart=200,
    #     parallel=true,
    # )

    term_cond = BOSS.IterLimit(iters)

    options = BOSS.BossOptions(;
        info=true,
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
