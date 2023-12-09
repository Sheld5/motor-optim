using BOSS
using Distributions
using JLD2
# using PRIMA
using OptimizationOptimJL, NLopt

include("motor_problem.jl")
include("data.jl")
include("ansys_model_final.jl")

function get_acquisition()
    BOSS.ExpectedImprovement(;
        cons_safe=true,
        ϵ_samples=1,
    )
end

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

get_param_objective() = (x) -> ModelParam.calc(x...)
get_ansys_objective() = load_ansys_model()

function get_problem(X, Y;
    surrogate_mode=:Semipar,  # :Semipar, :GP, :Param
)
    x_dim, y_dim = size(X)[1], size(Y)[1]

    objective = get_ansys_objective()
    domain = get_domain()
    θ_priors, length_scale_priors, noise_var_priors = get_priors(x_dim, y_dim)
    model = get_surrogate(Val(surrogate_mode), θ_priors, length_scale_priors)

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

function get_surrogate(::Val{:Param}, θ_priors, length_scale_priors)
    BOSS.NonlinModel(;
        predict = (x, θ) -> ModelParam.calc(x..., θ...),
        param_priors = θ_priors,
    )
end
function get_surrogate(::Val{:GP}, θ_priors, length_scale_priors)
    BOSS.Nonparametric(;
        length_scale_priors,
    )
end
function get_surrogate(::Val{:Semipar}, θ_priors, length_scale_priors)
    BOSS.Semiparametric(
        BOSS.NonlinModel(;
            predict = (x, θ) -> ModelParam.calc(x..., θ...),
            param_priors = θ_priors,
        ),
        BOSS.Nonparametric(;
            length_scale_priors,
        ),
    )
end

function get_model_fitter(::Val{:MLE}, surrogate_mode; parallel=true)
    ### PRIMA.jl causes `StackOverflowError` when parallelized on cluster.
    # BOSS.NewuoaMLE(PRIMA;
    #     multistart=1,#200,
    #     parallel,
    #     apply_softplus=true,
    #     softplus_params=get_softplus_params(surrogate_mode),
    #     rhoend=1e-3,
    # )
    BOSS.OptimizationMLE(;
        algorithm=NelderMead(),
        multistart=200,
        parallel,
        apply_softplus=true,
        softplus_params=get_softplus_params(surrogate_mode),
        x_tol=1e-3,
    )
end
function get_model_fitter(::Val{:BI}, surrogate_mode; parallel=true)
    BOSS.TuringBI(;
        sampler=BOSS.PG(20),  # TODO BOSS.NUTS(1000, 0.65) does not work. See https://github.com/TuringLang/DistributionsAD.jl/issues/260
        warmup=400,
        samples_in_chain=10,
        chain_count=8,
        leap_size=5,
        parallel,
    )
end
function get_model_fitter(::Val{:Random}, surrogate_mode; parallel=true)
    BOSS.RandomMLE()
end

get_softplus_params(::Val{:Param}) = fill(true, 3)
get_softplus_params(::Val{:Semipar}) = fill(true, 3)
get_softplus_params(::Val{:GP}) = nothing

function get_acq_maximizer(::Val{:Random}; parallel=true)
    BOSS.RandomSelectAM()
end
function get_acq_maximizer(::Val{:optim}; parallel=true)
    ### PRIMA.jl causes `StackOverflowError` when parallelized on cluster.
    # BOSS.CobylaAM(PRIMA;
    #     multistart=1,#200, # Make sure `multistart` >> 60 as Cobyla is not optimizing over the discrete `nk`.
    #     parallel,
    #     rhoend=1e-3,
    # )
    BOSS.NLoptAM(NLopt;
        algorithm=:LN_COBYLA,
        multistart=200, # Make sure `multistart` >> 60 as Cobyla is not optimizing over the discrete `nk`.
        parallel,
        xtol_abs=1e-3,
    )
end

function check_surrogate_mode(surrogate_mode, model)
    if (surrogate_mode == :Semipar)
        @assert model isa BOSS.Semiparametric
    elseif (surrogate_mode == :GP)
        @assert model isa BOSS.Nonparametric
    elseif (surrogate_mode == :Param)
        @assert model isa BOSS.NonlinModel
    else
        @assert false
    end
end

"""
Run BOSS on the motor problem.

# Keywords
- `init_data`: The number of randomly generated initial data points.
- `iters`: The number of iterations of BOSS.
- `model_fitter_mode`: Defines the model parameter estimation technique. Choose from `:MLE` and `:BI`.
- `acq_maximizer_mode`: Defines the acquisition function maximization technique. Choose from `:optim` and `:Random`.
- `surrogate_mode`: Defines the surrogate model used. Choose from `:Semipar` and `:GP`.
"""
function test_script(problem=nothing;
    init_data=1,
    iters=1,
    model_fitter_mode=:MLE,  # :MLE, :BI
    acq_maximizer_mode=:optim,  # :optim, :Random
    surrogate_mode=:Semipar,  # :Semipar, :GP, :Param
    parallel=true,
)
    if acq_maximizer_mode == :Random
        model_fitter_mode = :Random
    end
    if isnothing(problem)
        X, Y = get_data(init_data, get_domain())
        problem = get_problem(X, Y; surrogate_mode)
    else
        check_surrogate_mode(surrogate_mode, problem.model)
    end

    model_fitter = get_model_fitter(Val(model_fitter_mode), Val(surrogate_mode); parallel)
    acq_maximizer = get_acq_maximizer(Val(acq_maximizer_mode); parallel)
    acquisition = get_acquisition()
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
function runopt(;
    save_path="./data",
    init_data=2,
    runs=10,
    iters=10,
    id="",
    parallel=true,
)
    isempty(id) || (id *= "_")

    for r in 1:runs
        # new random data
        X, Y = get_data(init_data, get_domain())

        # Random
        problem = get_problem(deepcopy.((X, Y))...)
        res_rand = test_script(problem; iters, parallel, acq_maximizer_mode=:Random)
        
        # MLE
        problem = get_problem(deepcopy.((X, Y))...)
        res_mle = test_script(problem; iters, parallel)
        
        # BI
        problem = get_problem(deepcopy.((X, Y))...)
        res_bi = test_script(problem; iters, parallel, model_fitter_mode=:BI)
        
        # MLE - GP
        problem = get_problem(deepcopy.((X, Y))...; surrogate_mode=:GP)
        res_gp = test_script(problem; iters, parallel, surrogate_mode=:GP)
        
        # BI - GP
        problem = get_problem(deepcopy.((X, Y))...; surrogate_mode=:GP)
        res_gpbi = test_script(problem; iters, parallel, surrogate_mode=:GP, model_fitter_mode=:BI)

        # MLE - PARAM
        problem = get_problem(deepcopy.((X, Y))...; surrogate_mode=:Param)
        res_par = test_script(problem; iters, parallel, surrogate_mode=:Param)

        # BI - PARAM
        problem = get_problem(deepcopy.((X, Y))...; surrogate_mode=:Param)
        res_parbi = test_script(problem; iters, parallel, surrogate_mode=:Param, model_fitter_mode=:BI)

        # TODO
        save(save_path*"/rand_"*id*"$r.jld2", data_dict(res_rand.data))
        save(save_path*"/mle_"*id*"$r.jld2", data_dict(res_mle.data))
        save(save_path*"/bi_"*id*"$r.jld2", data_dict(res_bi.data))
        save(save_path*"/gp_"*id*"$r.jld2", data_dict(res_gp.data))
        save(save_path*"/gpbi_"*id*"$r.jld2", data_dict(res_gpbi.data))
        save(save_path*"/par_"*id*"$r.jld2", data_dict(res_par.data))
        save(save_path*"/parbi_"*id*"$r.jld2", data_dict(res_parbi.data))
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

#         problem = get_problem(deepcopy.((X, Y))...; surrogate_mode=:GP)
#         res = test_script(problem; iters, model_fitter_mode=:MLE, surrogate_mode=:GP)
#         save(dir*"/gp_$r.jld2", data_dict(res.data))
#     end
# end
