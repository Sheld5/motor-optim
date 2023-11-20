using PRIMA
using Distributed

include("../ansys_model_final.jl")

function test_pfor(; tasks=1)
    ansys = load_ansys_model()
    obj = (x) -> ansys(x) |> sum
    start = ModelParam.domain() |> mean

    best_x = fill(0., Threads.nthreads())
    res = fill(Inf, Threads.nthreads())

    Threads.@threads for i in 1:tasks
        x, info = newuoa(obj, start)
        val = obj(x)
        if val < res[Threads.threadid()]
            best_x[Threads.threadid()] = x
            res[Threads.threadid()] = val
        end
    end
    
    b = argmin(res)
    x = best_x[b]
    @show x
end

function test_ptask(; tasks=1)
    ansys = load_ansys_model()
    obj = (x) -> ansys(x) |> sum
    start = ModelParam.domain() |> mean

    starts = fill(start, tasks)
    ptasks = [Threads.@spawn newuoa(obj, s)[1] for s in starts]
    results = fetch.(ptasks)
    
    vals = obj.(results)
    best = argmin(vals)
    x = results[best]
    @show x
end

function test_dfor(; tasks=1)
    ansys = load_ansys_model()
    obj = (x) -> ansys(x) |> sum
    start = ModelParam.domain() |> mean

    starts = fill(start, tasks)
    best = @distributed (reduce_x(obj)) for i in 1:1
        newuoa(obj, start)[1]
    end
    
    @show best
end
function reduce_x(obj)
    function (xa, xb)
        if obj(xa) < obj(xb)
            return xa
        else
            return xb
        end
    end
end
