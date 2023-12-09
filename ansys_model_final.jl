using JLD2
using BOSS
using Distributions

include("data.jl")
include("Surrogate_Q_volne_parametry.jl")

# OPTIMUM:
# x = [46., 0.029, 0.49, 0.3]
# y = [129.94, -4.96]

const ANSYS_PARAMS_PATH = "/home/soldasim/motor-optim/motor-optim/ansys_model_params.jld2"

"""
Use `f = ansys_model_final()` to retrieve the ansys model.
"""
function ansys_model_final()
    x_dim = 4
    y_dim = 2
    domain = ModelParam.domain()

    return BOSS.Nonparametric(;
        length_scale_priors = fill(Product([truncated(Normal(0., dif/3); lower=0.) for dif in (domain[2][i]-domain[1][i] for i in 1:x_dim)]), y_dim),
    )
end

function save_ansys_params(data)
    save(ANSYS_PARAMS_PATH, data_dict(data))
end

function load_ansys_model()
    data = load_ansys_model_data()
    model = ansys_model_final()
    post = BOSS.model_posterior(model, data)
    return (x) -> post(x)[1]
end

function load_ansys_model_data()
    data_dict = load(ANSYS_PARAMS_PATH)
    return BOSS.ExperimentDataMLE(
        convert(Matrix{Float64}, data_dict["X"]),
        convert(Matrix{Float64}, data_dict["Y"]),
        data_dict["θ"],
        data_dict["length_scales"],
        data_dict["noise_vars"],
    )
end


# - - - TESTS - - - - -

function optimize_ansys()
    ansys = load_ansys_model()
    obj(x) = ansys([round(x[1]), x[2:4]...])[2]
    domain = ModelParam.domain()
    nonlinear_ineq(x) = obj(x)[1] - 600.

    multistart = 200
    starts = rand(4, multistart) .* (domain[2] - domain[1]) .+ domain[1]
    starts[1,:] .= round.(starts[1,:])

    best_x, best_y = nothing, Inf
    for s in eachcol(starts)
        x, info = cobyla(obj, s; xl=domain[1], xu=domain[2], nonlinear_ineq)
        y = obj(x)
        if y < best_y
            best_x, best_y = [round(x[1]), x[2:4]...], y
        end
    end
    return best_x, best_y
end
