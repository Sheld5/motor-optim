
ID = ARGS[1]
NAME = ARGS[2]

using Pkg
Pkg.activate("/home/soldasim/motor-optim/motor-optim")
# Pkg.resolve()
# Pkg.instantiate()
include("/home/soldasim/motor-optim/motor-optim/boss.jl")

t = @elapsed runopt(;
    save_path="/home/soldasim/motor-optim/motor-optim/experiments/data-$(NAME)",
    init_data=4,
    runs=2, #10
    iters=50,
    id=ID,
    parallel=true,
)

println()
@show t
