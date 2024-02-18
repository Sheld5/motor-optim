using BOSS
using Random
include("Surrogate_Q_volne_parametry.jl")

Random.seed!(555)

function get_domain()
    constraints(x) = [ModelParam.check_feas(x...)...]
    return BOSS.Domain(;
        bounds = ModelParam.domain(),
        discrete = ModelParam.discrete_dims(),
        cons = constraints,
    )
end

function grid(; save=false)
    lhc_size = 20
    bounds = ModelParam.domain()
    domain = get_domain()

    # generate X
    X = BOSS.generate_starts_LHC(bounds, lhc_size)
    for dim in 1:length(ModelParam.discrete_dims())
        ModelParam.discrete_dims()[dim] && (X[dim,:] .= round.(X[dim,:]))
    end

    # exclude exterior points
    X = reduce(hcat, (x for x in eachcol(X) if BOSS.in_domain(x, domain)))

    # calculate Y
    # Y = reduce(hcat, (ModelParam.calc(x...) for x in eachcol(X)))
    
    println("$(size(X)[2]) samples")
    save && save_grid(X)
    return X
end

function check_for_duplicates(X)
    xs = []
    duplicates = []
    for x in eachcol(X)
        if x in xs
            push!(duplicates, x)
        else
            push!(xs, x)
        end
    end
    return duplicates
end

function save_grid(X)
    open("./motor-optim/grid.txt", "w+") do io
        
        println(io)
        println(io, "\tnk\tdk\tDs\tQ")
        for x in eachcol(X)
            for n in x
                print(io, ' ', n)
            end
            println(io)
        end

        # println(io)
        # println(io, "\tdP\tTav")
        # for y in eachcol(Y)
        #     for n in y
        #         print(io, ' ', n)
        #     end
        #     println(io)
        # end

        println()
    end
end
