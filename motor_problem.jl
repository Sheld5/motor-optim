
# - - - - - - INITIAL DATA - - - - - -

include("Surrogate_Q_volne_parametry.jl")

function generate_data(LHC_size)
    domain = BOSS.Domain(;
        bounds = ModelParam.domain(),
        discrete = ModelParam.discrete_dims(),
        cons = (x)->ModelParam.check_feas(x...),
    )

    X, Y = generate_LHC(domain.bounds, LHC_size)
    X, Y = BOSS.exclude_exterior_points(domain, X, Y; info=false)
end

function generate_LHC(bounds, LHC_size)
    # generate some points
    X = BOSS.generate_starts_LHC(bounds, LHC_size)
    # round discrete dimensions
    for dim in 1:length(ModelParam.discrete_dims())
        ModelParam.discrete_dims()[dim] && (X[dim,:] .= round.(X[dim,:]))
    end
    # evaluate
    Y = reduce(hcat, (ModelParam.calc(x...) for x in eachcol(X)))

    return X, Y
end

function get_data()
    x = mean(ModelParam.domain())
    y = ModelParam.calc(x...)
    return hcat(x), hcat(y)
end


# - - - - - - PROBLEM DEFINITION - - - - - -

function example()
    # Initial data: (Corners and middle of the domain.)
    X, Y = generate_data()
    @show size(X), size(Y)

    x = X[:,1]; y = Y[:,1]

    # Objective function: (minimize Tav)
    @show y ≈ ModelParam.calc(x...)
    println("minimize $(y[2])")

    # Box-constraints on x:
    bounds = ModelParam.domain()
    all(bounds[1] .<= x .<= bounds[2]) ? println("$x is in bounds") : println("$x out of bounds")

    # Non-linear constraints on x: (constructability)
    all(ModelParam.check_feas(x...) .>= 0.) ? println("$x is feasible") : println("$x is infeasible")

    # Constraints on y: (dP <= 300.)
    all(y .<= ModelParam.y_max()) ? println("$y is feasible") : println("$y is infeasible")

    # Discrete dimensions: (nk je diskretni)
    @show ModelParam.discrete_dims()
end
