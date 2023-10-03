
# using Plots
using BOSS
using JLD2

include("Surrogate_Q_volne_parametry.jl")


# - - - - - - MANUAL MODEL ANALYSIS - - - - - -

function test_corners(domain, func)
    @assert length(domain) == 2
    @assert length(domain[1]) == length(domain[2])
    dim = length(domain[1])
    coors = [Tuple(i) for i in CartesianIndices(Tuple(2 for _ in 1:dim))]
    corners = [Tuple(domain[coor[i]][i] for i in 1:dim) for coor in coors]
    return [(corner, func(corner...)) for corner in corners]
end

# domain_eliptic() = vcat(MotorParam.domain()[1], 0.), vcat(MotorParam.domain()[2], 1.)
# calc_eliptic(nk, dk, Ds, a) = MotorParam.calc(nk, dk, Ds; a)
# calc_Q(nk, dk, Ds, a, Q) = MotorParam.calc(nk, dk, Ds; a, Q)



function plot_nk(dk=0.019, Ds=0.410, Q=0.03)
    nk_range = (30, 60)
    
    dP_vals = []
    Tav_vals = []
    for nk in range(nk_range...)
        dP, Tav, S = ModelParam.calc(nk, dk, Ds, Q)
        push!(dP_vals, dP)
        push!(Tav_vals, Tav)
    end

    scatter(dP_vals, Tav_vals; title="nk=30,...,60 | dk=$dk | Ds=$Ds", xlabel="dP", ylabel="Tav")
end

function plot_dk(nk=60, Ds=0.410, Q=0.03)
    dk_range = (0.010, 0.030)
    points = 200
    
    dP_vals = []
    Tav_vals = []
    for dk in range(dk_range...; length=points)
        dP, Tav, S = ModelParam.calc(nk, dk, Ds, Q)
        push!(dP_vals, dP)
        push!(Tav_vals, Tav)
    end

    scatter(dP_vals, Tav_vals; title="nk=$nk | dk=$(dk_range[1]),...,$(dk_range[2]) | Ds=$Ds", xlabel="dP", ylabel="Tav")
end

function plot_Ds(nk=60, dk=0.019, Q=0.03)
    Ds_range = (0.410, 0.520)
    points = 200
    
    dP_vals = []
    Tav_vals = []
    for Ds in range(Ds_range...; length=points)
        dP, Tav, S = ModelParam.calc(nk, dk, Ds, Q)
        push!(dP_vals, dP)
        push!(Tav_vals, Tav)
    end

    scatter(dP_vals, Tav_vals; title="nk=$nk | dk=$dk | Ds=$(Ds_range[1]),...,$(Ds_range[2])", xlabel="dP", ylabel="Tav")
end

function plot_a(nk=30, dk=0.0302, Ds=0.359; zoom=true)
    a_range = (0., 1.)
    points = 2000
    
    dP_vals_good = []
    Tav_vals_good = []
    dP_vals_bad = []
    Tav_vals_bad = []

    for a in range(a_range...; length=points)
        dP, Tav, S = calc_eliptic(nk, dk, Ds, a)
        if all(ModelParam.domain_constraints(nk, dk, Ds, a) .< 0)
            push!(dP_vals_good, dP)
            push!(Tav_vals_good, Tav)
        else
            push!(dP_vals_bad, dP)
            push!(Tav_vals_bad, Tav)
        end
    end

    isempty(dP_vals_good) && println("no feasible points")
    if zoom
        xlims = isempty(dP_vals_good) ? :nothing : extrema(dP_vals_good)
    else
        xlims = :nothing
    end

    scatter(; xlims, title="nk=$nk | dk=$dk | Ds=$Ds | a=$(a_range[1]),...,$(a_range[2])", xlabel="dP", ylabel="Tav")
    scatter!(dP_vals_good, Tav_vals_good; color=:green, label="feasible")
    scatter!(dP_vals_bad, Tav_vals_bad; color=:red, label="infeasible")
end

function plot_Q(nk=38, dk=0.02, Ds=0.36, a=0.014; zoom=true)
    Q_range = (0.2, 0.9)
    points = 200
    
    dP_vals_good = []
    Tav_vals_good = []
    dP_vals_bad = []
    Tav_vals_bad = []

    for Q in range(Q_range...; length=points)
        dP, Tav, S = calc_Q(nk, dk, Ds, a, Q)
        if all(ModelParam.domain_constraints(nk, dk, Ds, a) .< 0)
            push!(dP_vals_good, dP)
            push!(Tav_vals_good, Tav)
        else
            push!(dP_vals_bad, dP)
            push!(Tav_vals_bad, Tav)
        end
    end

    isempty(dP_vals_good) && println("no feasible points")
    if zoom
        xlims = isempty(dP_vals_good) ? :nothing : extrema(dP_vals_good)
    else
        xlims = :nothing
    end

    scatter(; xlims, title="nk=$nk | dk=$dk | Ds=$Ds | a=$a\nQ=$(Q_range[1]),...,$(Q_range[2])", xlabel="dP", ylabel="Tav")
    scatter!(dP_vals_good, Tav_vals_good; color=:green, label="feasible")
    scatter!(dP_vals_bad, Tav_vals_bad; color=:red, label="infeasible")
end

function plot_feasibility(Ds=0.410, a=0.; zoom=true, make_plot=true)
    nk_range = (30, 60)
    dk_range = (0.010, 0.030)
    # nk_range = (1,100)
    # dk_range = (0.0001,1.)
    points = 200

    dP_vals_good = []
    Tav_vals_good = []
    dP_vals_bad = []
    Tav_vals_bad = []

    for nk in range(nk_range...)
        for dk in range(dk_range...; length=points)
            dP, Tav, S = ModelParam.calc(nk, dk, Ds, Q)
            if all(ModelParam.domain_constraints(nk, dk, Ds, a) .< 0)
                push!(dP_vals_good, dP)
                push!(Tav_vals_good, Tav)
            else
                push!(dP_vals_bad, dP)
                push!(Tav_vals_bad, Tav)
            end
        end
    end

    make_plot && isempty(dP_vals_good) && println("no feasible points")
    if zoom
        xlims = isempty(dP_vals_good) ? :nothing : extrema(dP_vals_good)
    else
        xlims = :nothing
    end

    if make_plot
        scatter(; title="nk=$(nk_range[1]),...,$(nk_range[2]) | dk=$(dk_range[1]),...,$(dk_range[2]) | Ds=$Ds | a=$a", xlabel="dP", ylabel="Tav")
        scatter!(dP_vals_good, Tav_vals_good; color=:green, label="feasible")
        scatter!(dP_vals_bad, Tav_vals_bad; color=:red, label="infeasible")
    end
    
    return make_plot ? scatter!() : !isempty(dP_vals_good)
end

# function test_feasibility()
#     Ds_range = (0.410,0.520)
#     # Ds_range = (0.298,0.399)
#     points = 200
#     feas = [plot_feasibility(Ds; make_plot=false) for Ds in range(Ds_range...; length=points)]
#     return any(feas)
# end

function test_feasibility(; points=200)
    domain = ModelParam.domain()
    nk_range = range(domain[1][1], domain[2][1])
    dk_range = range(domain[1][2], domain[2][2]; length=points)
    Ds_range = range(domain[1][3], domain[2][3]; length=points)
    # Q_range = range(domain[1][4], domain[2][4]; length=points)
    
    for nk in nk_range
        for dk in dk_range
            for Ds in Ds_range
                ModelParam.check_feas(nk, dk, Ds) && (return true)
            end
        end
    end
    return false
end



# - - - - - - PARAM MODEL RESIDUALS TO ANSYS - - - - - -

function ansys_residuals()
    X, Y = get_old_data()
    problem = get_problem_param_model(X, Y)

    model_fitter = BOSS.OptimizationMLE(;
        algorithm=LBFGS(),
        multistart=200,
        parallel=true,
        apply_softplus=true,
        softplus_params=fill(true, 3),
        x_tol=1e-3,

    )

    options = BOSS.BossOptions(;
        info=true,
        ϵ_samples=1,
    )

    BOSS.estimate_parameters!(problem, model_fitter; options)
    posterior = BOSS.model_posterior(problem.model, problem.data)

    Y_hat = reduce(hcat, (posterior(x)[1] for x in eachcol(X)))
    X, Y, Y_hat
end

function plot_residuals_dptav(X, Y, Y_hat)
    nk = X[1,:]
    Dk = X[2,:]
    residuals = Y_hat .- Y
    @show minimum(residuals[1,:]), maximum(residuals[1,:])
    @show minimum(residuals[2,:]), maximum(residuals[2,:])

    xlabel = "nk"
    ylabel = "Dk"

    # 2D
    p1 = scatter(nk, Dk; xlabel, ylabel, zlabel="dP", zcolor=residuals[1,:] ./ 1000., title="dp / 1000")
    p2 = scatter(nk, Dk; xlabel, ylabel, zlabel="Tav", zcolor=residuals[2,:], title="Tav")
    # 3D
    # p1 = scatter(nk, Dk, residuals[1,:]; xlabel, ylabel, zlabel="dP", zcolor=residuals[1,:])
    # p2 = scatter(nk, Dk, residuals[2,:]; xlabel, ylabel, zlabel="Tav", zcolor=residuals[2,:])
    
    display.((p1, p2))
end

function plot_residuals_dkds(X, Y, Y_hat)
    Dk = Float64[]
    ds = Float64[]
    dP_res = Float64[]
    Tav_res = Float64[]

    for (x, y, y_hat) in zip(eachcol(X), eachcol(Y), eachcol(Y_hat))
        if x[1] == 60.
            push!(Dk, x[2]); push!(ds, x[3]);
            push!(dP_res, y_hat[1]-y[1]); push!(Tav_res, y_hat[2]-y[2]);
        end
    end

    @show minimum(dP_res), maximum(dP_res)
    @show minimum(Tav_res), maximum(Tav_res)

    println("\nPOINTS:  ds, Dk | dP_diff, Tav_diff")
    for (ds_, Dk_, dP_, Tav_) in zip(ds, Dk, dP_res, Tav_res)
        println("$ds_ $Dk_ | $dP_, $Tav_")
    end

    xlabel = "Dk"
    ylabel = "ds"

    # 2D
    p1 = scatter(Dk, ds; xlabel, ylabel, zlabel="dP", zcolor=dP_res ./ 1000., title="dp / 1000  (nk=60)")
    p2 = scatter(Dk, ds; xlabel, ylabel, zlabel="Tav", zcolor=Tav_res, title="Tav  (nk=60)")
    # 3D
    # p1 = scatter(Dk, ds, dP; xlabel, ylabel, zlabel="dP", zcolor=dP)
    # p2 = scatter(Dk, ds, Tav; xlabel, ylabel, zlabel="Tav", zcolor=Tav)
    
    display.((p1, p2))
end

function get_problem_param_model(X, Y)
    x_dim, y_dim = size(X)[1], size(Y)[1]

    objective(x) = ModelParam.calc(x...)
    parametric(x, θ) = ModelParam.calc(x..., θ...)
    constraints(x) = ModelParam.check_feas(x...)

    domain = BOSS.Domain(;
        bounds = ModelParam.domain(),
        discrete = ModelParam.discrete_dims(),
        cons = constraints,
    )
    
    θ_priors, _, noise_var_priors = get_priors(x_dim, y_dim)

    model = BOSS.NonlinModel(;
        predict=parametric,
        param_priors=θ_priors,
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

# init_data = 4
function compare_mle_bi(init_data)
    dir = "./data"
    fitness = BOSS.LinFitness([0., -1.])
    y_max = ModelParam.y_max()

    files = readdir(dir)
    mle = [load(dir*"/"*f) for f in files if startswith(f, "mle")]
    bi = [load(dir*"/"*f) for f in files if startswith(f, "bi")]

    mle_bsf = bsf_series.(mle, Ref(fitness), Ref(y_max), Ref(init_data))
    bi_bsf = bsf_series.(bi, Ref(fitness), Ref(y_max), Ref(init_data))

    plot(; title="MLE vs BI | median,min,max fitness", ylabel="-Tav", xlabel="iteration")
    plot_runs!(mle_bsf; label="MLE")
    plot_runs!(bi_bsf; label="BI")
end

function plot_runs!(runs; label=nothing)
    # assert all runs have the same number of iterations
    iterations = first(runs)[1]
    @assert all(r -> r[1]==iterations, runs)

    mins = minimum.(((r[2][i] for r in runs) for i in 1:length(iterations)))
    maxs = maximum.(((r[2][i] for r in runs) for i in 1:length(iterations)))
    meds = median.(((r[2][i] for r in runs) for i in 1:length(iterations)))

    plot!(iterations, meds; yerror=(meds.-mins, maxs.-meds), label)
end

function bsf_series(res, fitness, y_max, init_data)
    feasible(y) = all(y .<= y_max)
    X, Y = res["X"], res["Y"]
    @assert size(X)[2] == size(Y)[2]

    iters = size(X)[2] - init_data
    iteration = [i for i in 0:iters]

    bsf = [maximum((fitness(y) for y in eachcol(Y[:,1:init_data]) if feasible(y)))]
    for i in 1:iters
        y = Y[:,init_data+i]
        if feasible(y) && (fitness(y) > last(bsf))
            push!(bsf, fitness(y))
        else
            push!(bsf, last(bsf))
        end
    end

    return iteration, bsf
end



# - - - - - - OLD DATA - - - - - -

function get_old_data()
    Q = 0.3
    #   nk      dk      Ds      dP      Tav     Pfe
    data = [
        # init data
        34  	25.0	474.2	768.1	123.0	1460.8
        42  	17.0	481.5	2782.3	104.5	1582.7
        40  	11.7	433.8	15541.5	84.0	1672.1
        30  	19.0	485.2	3507.1	114.1	1600.1
        46  	16.3	444.8	2731.0	94.3	1580.9
        32  	23.7	437.5	1119.7	113.5	1505.3
        42  	10.3	477.8	24373.9	87.3	1685.0
        50  	13.0	415.5	6245.2	82.2	1632.0
        58  	15.0	441.2	2498.1	88.2	1570.5
        54  	19.7	470.5	854.5	105.4	1465.7
        44  	23.0	455.8	656.6	110.9	1433.8
        38  	27.7	430.2	375.1	115.9	1356.1
        32  	15.7	452.2	6945.7	98.8	1639.9
        34  	12.3	496.2	17703.5	99.0	1675.8
        40  	29.7	466.8	235.6	131.3	1274.3
        36  	17.7	419.2	3220.0	97.0	1594.7
        52  	11.0	463.2	11869.6	84.0	1660.8
        44  	21.7	411.8	858.7	99.3	1468.8
        56  	14.3	492.5	3275.6	97.9	1591.1
        # non-param run
        60      18.0    505.5   971     107     1163
        60      18.3    518.5   898     110.1   1085
        60      21.9    416.4   379.6   98.3    1922   
        60      18.0    435.7   971     93.6    1567
        60      18.0    433.3   971     93.2    1580
        60      18.8    429.2   971     92.4    1601
        60      18.0    435.6   971     93.6    1568
        60      18.0    435.9   971     93.7    1566    
        60      18.0    435.8   971     93.6    1566
        60      18.0    435.7   971     93.6    1567
        60      18.0    435.8   971     93.6    1566
        60      18.0    506.2   971     107.1   1159
        30      18.0    410.0   4181    96.4    1393
        60      18.2    411.2   971     96.8    1693
        60      18.0    435.9   971     100.1   1566
        60      18.3    416.3   898     90      1685
        60      18.0    514.5   971     107.8   1108
        60      18.8    411.3   791     90.2    1742
        60      29.7    520.0   45      133.2   1159
        60      21.1    422.5   456     97      1524         
        59      18.1    412.4   984     89.2    1685     
        60      18.0    436.0   971     93.1    1566
        31      18.6    518.7   3409    117     820
        60      18.0    436.1   971     93.2    1565  
        60      21.7    507.0   397     114.1   1204
        60      18.0    435.9   971     93.1    1566
        60      18.1    411.2   946     88.7    1699     
        60      18.0    435.9   971     93.1    1566  
        60      18.0    435.9   971     93.1    1566 
        60      18.4    412.7   875     89.6    1790  
        33      30.0    410.0   459     120     1932               
        60      28.6    520.0   69      130.9   1149  
        46      23.3    410.0   549     102     1893    
        48      18.0    410.0   1600    91.3    1600  
        60      18.0    435.9   971     100.1   1566 
        60      18.0    436.4   971     93.2    1564
        60      18.6    410.6   832     89.6    1733    
        60      18.4    413.4   875     89.7    1706   
        60      21.8    415.0   388     95.3    983  
    ]

    X = data[:,1:3]'
    X = vcat(X, fill(Q, size(X)[2])')
    X[2:3,:] ./= 1000.

    Y = data[:,4:5]'

    domain = BOSS.Domain(;
        bounds = ModelParam.domain(),
        discrete = ModelParam.discrete_dims(),
        cons = (x)->ModelParam.check_feas(x...),
    )

    Xf = Vector{Float64}[]
    Yf = Vector{Float64}[]
    for (x,y) in zip(eachcol(X), eachcol(Y))
        if BOSS.in_domain(domain, x)
            push!(Xf, x)
            push!(Yf, y)
        end
    end

    return reduce(hcat, Xf), reduce(hcat, Yf)
end
