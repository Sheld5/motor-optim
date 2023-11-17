using BOSS
using Distributions
using OptimizationOptimJL
using XLSX

include("motor_problem.jl")


# - - - - - - - - MODELS - - - - - - - -

# include("fit_grid_nn.jl")

ansys_model_poly2() = BOSS.NonlinModel(;
    predict = (x, θ) -> [
        θ[1] + θ[2]*x[1] + θ[3]*x[2] + θ[4]*x[1]^2 + θ[5]*x[1]*x[2] + θ[6]*x[2]^2,
        θ[7] + θ[8]*x[1] + θ[9]*x[2] + θ[10]*x[1]^2 + θ[11]*x[1]*x[2] + θ[12]*x[2]^2,
    ],
    param_priors = fill(Normal(0., 100.), 12),
)
ansys_model_poly3() = BOSS.NonlinModel(;
    predict = (x, θ) -> [
        θ[1] + θ[2]*x[1] + θ[3]*x[2] + θ[4]*x[1]^2 + θ[5]*x[1]*x[2] + θ[6]*x[2]^2 + θ[7]*x[1]^3 + θ[8]*x[1]^2*x[2] + θ[9]*x[1]*x[2]^2 + θ[10]*x[2]^3,
        θ[11] + θ[12]*x[1] + θ[13]*x[2] + θ[14]*x[1]^2 + θ[15]*x[1]*x[2] + θ[16]*x[2]^2 + θ[17]*x[1]^3 + θ[18]*x[1]^2*x[2] + θ[19]*x[1]*x[2]^2 + θ[20]*x[2]^3,
    ],
    param_priors = fill(Normal(0., 100.), 20),
)

function ansys_model_param()
    param_range = ModelParam.param_range()

    BOSS.NonlinModel(;
        predict = (x, θ) -> ModelParam.calc(x..., θ...),
        param_priors = [truncated(Normal(mid, dif/3); lower=0.) for (mid, dif) in zip(mean(param_range), param_range[2]-param_range[1])],
    ) 
end

function ansys_model_gp()
    x_dim = 4
    y_dim = 2
    domain = ModelParam.domain()

    BOSS.Nonparametric(;
        length_scale_priors = fill(Product([truncated(Normal(0., dif/3); lower=0.) for dif in (domain[2][i]-domain[1][i] for i in 1:x_dim)]), y_dim),
    )
end

function ansys_model_semipar()
    x_dim = 4
    y_dim = 2
    param_range = ModelParam.param_range()
    domain = ModelParam.domain()

    BOSS.Semiparametric(
        BOSS.NonlinModel(;
            predict = (x, θ) -> ModelParam.calc(x..., θ...),
            param_priors = [truncated(Normal(mid, dif/3); lower=0.) for (mid, dif) in zip(mean(param_range), param_range[2]-param_range[1])],
        ),
        BOSS.Nonparametric(;
            length_scale_priors = fill(Product([truncated(Normal(0., dif/3); lower=0.) for dif in (domain[2][i]-domain[1][i] for i in 1:x_dim)]), y_dim),
        ),
    )
end

ansys_noise_var_priors() = fill(Dirac(1e-8), 2)
# ansys_noise_var_priors() = [truncated(Normal(0., 1/3); lower=0.), truncated(Normal(0., 0.01/3); lower=0.)]


# - - - - - - - - SCRIPTS - - - - - - - -

function load_grid()
    path = "./motor-optim/Vysledky_Ansys.xlsx"
    sheet = XLSX.readxlsx(path)["List3"];
    D = convert(Matrix{Float64}, sheet["A2:F55"])
    X, Y = D[:, 1:4]', D[:, 5:6]'
    return X, Y
end
# function load_grid_old()
#     path = "./motor-optim/Grid_final_Ansys_noduplicates.txt"
#     a = open(path) do f
#         lines = collect(eachline(f))[2:end]
#         reduce(hcat, (l -> parse.(Float64, split(l))).(lines))
#     end
#     X, Y = a[1:4, :], a[5:6, :]
#
#     # swap dk, Ds back
#     dk = X[3,:]
#     Ds = X[2,:]
#     X[2,:] = dk
#     X[3,:] = Ds
#
#     return X, Y
# end

function fit_grid(X, Y)
    problem = BOSS.OptimizationProblem(;
        fitness = BOSS.LinFitness([0., -1.]),  # irrelevant
        f = x->nothing,  # irrelevant
        domain = get_domain(),  # irrelevant
        y_max = ModelParam.y_max(),  # irrelevant
        model = ansys_model_gp(),
        noise_var_priors = ansys_noise_var_priors(),
        data = BOSS.ExperimentDataPrior(X, Y),
    )

    model_fitter = BOSS.OptimizationMLE(;
        algorithm=LBFGS(),
        multistart=200,
        parallel=true,
        apply_softplus=true,
        x_tol=1e-3,
    )

    options = BOSS.BossOptions()

    BOSS.estimate_parameters!(problem, model_fitter; options)
    post = BOSS.model_posterior(problem.model, problem.data)
    return problem.data, post
end

function eval_mse(post, X, Y)
    mse = (post, x, y) -> ((post(x)[1] .- y).^2)
    errs = mse.(Ref(post), eachcol(X), eachcol(Y))
    return sqrt.(mean(errs))
end

function print_errs(post, X, Y)
    for (x, y) in zip(eachcol(X), eachcol(Y))
        yhat = post(x)[1]
        r(n) = round(n; digits=3)
        println(y, " - ", r.(yhat), " = ", r.(abs.(y - yhat)))
    end
end

function crosscheck(runs=10)
    X, Y = load_grid()
    @show size(X), size(Y)
    data_size = size(X)[2]

    errs = []
    for _ in 1:runs
        test = sample(1:data_size, Int(round(data_size/10)); replace=false)
        train = [i for i in 1:data_size if !(i in test)]

        params, post = fit_grid(X[:,train], Y[:,train]);
        print_errs(post, X[:,test], Y[:,test])
        push!(errs, eval_mse(post, X[:,test], Y[:,test]))
    end
    @show mean(errs)
    return errs
end
