
# - - - - - - INITIAL DATA - - - - - -

include("Surrogate_Q_volne_parametry.jl")

function get_domain()
    constraints(x) = [ModelParam.check_feas(x...)...]
    return BOSS.Domain(;
        bounds = ModelParam.domain(),
        discrete = ModelParam.discrete_dims(),
        cons = constraints,
    )
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

function get_data(len::Int, domain::BOSS.Domain)
    X = reduce(hcat, (rand_interior_point(domain) for _ in 1:len))
    Y = reduce(hcat, (ModelParam.calc(x...) for x in eachcol(X)))
    return X[:,:], Y[:,:]
end

function rand_interior_point(domain::BOSS.Domain)
    x = BOSS.random_start(domain.bounds)
    x = BOSS.cond_func(round).(domain.discrete, x)
    if !isnothing(domain.cons)
        while any(domain.cons(x) .< 0.)
            x = BOSS.random_start(domain.bounds)
            x = BOSS.cond_func(round).(domain.discrete, x)
        end
    end
    return x
end


# - - - - - - PROBLEM DEFINITION - - - - - -

function example()
    # Initial data: (Corners and middle of the domain.)
    X, Y = generate_data(4, get_domain())
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
