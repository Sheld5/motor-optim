
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
    X = [
        100.0  0.011  0.482892  0.55
        74.0  0.005  0.495865  0.218
        37.0  0.01   0.465811  0.491
        68.0  0.003  0.467432  0.184
        83.0  0.007  0.424622  0.735
         9.0  0.012  0.429811  0.204
        63.0  0.006  0.486459  0.806
         8.0  0.026  0.477054  0.771
        37.0  0.024  0.476405  0.532
        11.0  0.018  0.435649  0.636
        31.0  0.03   0.466351  0.604
        88.0  0.009  0.454568  0.199
        72.0  0.008  0.432405  0.321
         8.0  0.004  0.472405  0.862
        18.0  0.001  0.492514  0.512
        69.0  0.002  0.483216  0.49 
        40.0  0.025  0.475541  0.437
        17.0  0.037  0.472189  0.4 
    ]
    Y = [
        4119.36         49.3544
        46160.1          55.0678
        35443.8          52.775
       505075.0          46.9722
        82986.2          41.2988
        44215.8          93.7309
            3.56897e5    40.1269
        29098.3          73.2465
          926.0          68.646
        49423.3          62.4764
          649.581        77.981
         1736.69         70.1366
        11343.2          55.123
            1.83811e8    39.5814
            1.97212e10   34.0769
            2.89121e7    36.395
          446.032        76.6124
          402.151       113.792
    ]
    return X'[:,:], Y'[:,:]
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
