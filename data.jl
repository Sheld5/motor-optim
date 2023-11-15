"""
Convert `BOSS.ExperimentDataPost` to a disctionary better suited for serialization into files.
"""
function data_dict(data::BOSS.ExperimentDataPost)
    return Dict(
        "X"=>data.X,
        "Y"=>data.Y,
        "θ"=>data.θ,
        "length_scales"=>data.length_scales,
        "noise_vars"=>data.noise_vars,
    )
end
