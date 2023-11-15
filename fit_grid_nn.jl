# This will prompt if neccessary to install everything, including CUDA:
using Flux, Statistics, ProgressMeter
# using CUDA  # for GPU

function train_ansys_nn(X, Y)
    X = convert(Matrix{Float32}, X)
    Y = convert(Matrix{Float32}, Y)
    Y[1,:] ./= 10_000.
    Y[2,:] ./= 100.

    model = Chain(
        BatchNorm(4),
        Dense(4 => 12, relu),
        BatchNorm(12),
        Dropout(0.4),
        Dense(12 => 2),
    )
    loss(model, X, Y) = mean(abs2.(model(X) .- Y))
    opt = Flux.setup(Flux.Adam(0.01), model)
    data = [(X, Y)]

    @show loss(model, X, Y)
    @showprogress for epoch in 1:10_000
        Flux.train!(loss, model, data, opt)
    end
    @show loss(model, X, Y)

    return model
end

function nn_error(model, X, Y)
    Yhat = model(X)
    Yhat[1,:] .*= 10_000.
    Yhat[2,:] .*= 100.
    err = [sqrt(mean(abs2.(Yhat[1,:] .- Y[1,:]))), sqrt(mean(abs2.(Yhat[2,:] .- Y[2,:])))]
end
