using Pkg
Pkg.activate(".")

ID = ARGS[1]

include("../boss.jl")

t = @elapsed runopt(;
    save_path="./experiments/data",
    init_data=4,
    runs=10,
    iters=50,
    id=ID,
    parallel=false,  # Parallelization causes `StackOverflowError` on RCI cluster.
)

println()
@show t
