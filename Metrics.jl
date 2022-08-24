module Metrics

using Pipe: @pipe
using Statistics: mean

function prediction_to_onehot(ŷ::Array{Float32,4})
    rows, cols, _, obs = size(ŷ)
    onehot = zeros(Float32, size(ŷ))
    for row in 1:rows, col in 1:cols, ob in 1:obs
        index = argmax(ŷ[row,col,:,ob])
        onehot[row,col,index,ob] = 1.0f0
    end
    return onehot
end

function true_positives(ŷ::Array{Float32,4}, y::Array{Float32,4})
    return @pipe ŷ .* y |> sum(_, dims=(1, 2, 4)) |> reshape(_, size(y)[3])
end

function false_positives(ŷ::Array{Float32,4}, y::Array{Float32,4})
    return @pipe ŷ .* (1 .- y) |> sum(_, dims=(1, 2, 4)) |> reshape(_, size(y)[3])
end

function false_negatives(ŷ::Array{Float32,4}, y::Array{Float32,4})
    return @pipe (1 .- ŷ) .* y |> sum(_, dims=(1, 2, 4)) |> reshape(_, size(y)[3])
end

function precision(ŷ::Array{Float32,4}, y::Array{Float32,4})
    ŷ = prediction_to_onehot(ŷ)
    tp = true_positives(ŷ, y)
    fp = false_positives(ŷ, y)
    return tp ./ (tp .+ fp)
end

function precision(ŷ::AbstractArray{Float32,4}, y::AbstractArray{Float32,4})
    return precision(Array(ŷ), Array(y))
end

function precision(ŷ::Array{Float32,4}, y::Array{Float32,4}, agg::Function)
    return precision(ŷ, y) |> agg
end

function recall(ŷ::Array{Float32,4}, y::Array{Float32,4})
    ŷ = prediction_to_onehot(ŷ)
    tp = true_positives(ŷ, y)
    fn = false_negatives(ŷ, y)
    return tp ./ (tp .+ fn)
end

function recall(ŷ::AbstractArray{Float32,4}, y::AbstractArray{Float32,4})
    return recall(Array(ŷ), Array(y))
end

function recall(ŷ::Array{Float32,4}, y::Array{Float32,4}, agg::Function)
    return recall(ŷ, y) |> agg
end

function IoU(ŷ::AbstractArray{Float32,4}, y::AbstractArray{Float32,4})
    return IoU(Array(ŷ), Array(y))
end

function IoU(ŷ::Array{Float32,4}, y::Array{Float32,4})
    ϵ = eps(Float32)
    ŷ = prediction_to_onehot(ŷ)
    tp = true_positives(ŷ, y)
    fn = false_negatives(ŷ, y)
    fp = false_positives(ŷ, y)
    return (tp .+ ϵ) ./ (tp .+ fp .+ fn .+ ϵ)
end

function mIoU(ŷ::Array{Float32,4}, y::Array{Float32,4})
    return IoU(ŷ, y) |> mean
end

end