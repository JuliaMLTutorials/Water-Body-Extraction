using Plots, Images, ArchGDAL, Flux, MLUtils, CUDA, Random, ProgressBars, Statistics
using Pipe: @pipe

include("UNet.jl")
include("Pipeline.jl")

const BATCH_SIZE = 2
const η = 1e-4

function dice_loss(ŷ::AbstractArray{Float32, 4}, y::AbstractArray{Float32, 4})
    ϵ = eps(Float32)
    intersection = sum(ŷ .* y, dims=(1, 2, 4))
    union = sum(ŷ, dims=(1, 2, 4)) .+ sum(y, dims=(1, 2, 4))
    dice_coefficient = ((2.0f0 .* intersection) .+ ϵ) ./ (union .+ ϵ)
    return mean(1.0f0 .- dice_coefficient)
end

function load_data(at::AbstractFloat)
    # List All Tiles
    sample_tiles = [114, 236, 318, 669, 676, 790, 991]
    tiles = collect(1:1600)
    filter!(x -> !(x in sample_tiles), tiles)
    shuffle!(tiles)

    # Split Tiles Into Training And Test
    split_index = 1600 * at |> floor |> Int
    train = ImagePipeline(tiles[1:split_index])
    test = ImagePipeline(tiles[split_index+1:end])

    # Construct DataLoaders
    train_data = DataLoader(train, batchsize=BATCH_SIZE, shuffle=true)
    test_data = DataLoader(test, batchsize=1, shuffle=false)
    return train_data, test_data
end

function get_model()
    UNetModule.UNet(5, 2) |> gpu
end

function train_model(model, train_data)
    # Define Optimizer
    opt = Flux.Optimiser(ClipNorm(η), Adam(η))

    # Get Parameters
    params = Flux.params(model)

    # Define Loss
    loss(x, y) = dice_loss(model(x), y)

    # Train For Two Epochs
    for epoch in 1:2

        l = 0.0f0
        iter = ProgressBar(train_data)
        for (i, (x, y)) in enumerate(iter)

            water_content = @pipe sum(y[:,:,2,:]) |> /(_, 512 * 512) |> *(_, 100)

            if water_content > 0.0

                # Compute Gradients
                grads = Flux.gradient(() -> loss(x, y), params)

                # Update Parameters
                Flux.Optimise.update!(opt, params, grads)

                current_loss = loss(x, y)
                l += current_loss
                set_description(iter, "Loss: $(round(current_loss, digits=4, base=10)), Average Loss: $(round(l / Float32(i), digits=4, base=10))")
        
            end
        end

        evaluate_model(model, "prediction_$epoch.png")

    end
end

function showimg(img::Any, sz::Tuple{Int,Int})
    plot(img, size=sz, axis=nothing, showaxis=false, margin=0Plots.mm)
end

function showimg(img::AbstractVector, sz::Tuple{Int,Int}, layout::Tuple{Int,Int})
    plot(img..., layout=layout, size=sz, axis=nothing, showaxis=false, margin=0Plots.mm)
end

function plot_color(img::Array{Float32,3}, gamma=1.0)
    scale = @pipe findmax(img)[1] |> Float32 |> max(_, 1.0f0)
    @pipe img .|> 
    /(_, scale) |>
    permutedims(_, (3, 2, 1)) |> 
    colorview(RGB, _) |> 
    adjust_histogram(_, GammaCorrection(gamma=gamma)) |>
    showimg(_, (1000, 1000))
end;

function plot_gray(img::Array{Float32,3})
    scale = @pipe findmax(img)[1] |> Float32 |> max(_, 1.0f0)
    @pipe img[:,:,1] |> 
    /(_, scale) |>
    permutedims(_, (2, 1)) |> 
    colorview(Gray, _) |>
    showimg(_, (1000, 1000))
end;

function prediction_to_mask(ŷ::Array{Float32,4})
    (mapslices(argmax, ŷ, dims=3) .|> Float32) .- 1.0f0
end

function show_prediction(model, tile::Int)
    # Plot Features
    rgb_plot = read_rgb(tile) |> plot_color
    nir_plot = read_nir(tile) |> plot_gray
    swir_plot = read_swir(tile) |> plot_gray
    mask_plot = read_mask(tile) |> plot_gray
    
    # Plot Prediction
    x, _ = ImagePipeline([tile])[1]
    prediction_plot = @pipe model(x) |> Array |> prediction_to_mask(_)[:,:,:,1] |> plot_gray

    # Plot Features And Prediction
    showimg([rgb_plot, nir_plot, swir_plot, mask_plot, prediction_plot], (5000, 1000), (1, 5))
end

function evaluate_model(model, filename="prediction.png")
    tiles = [114, 236, 318, 669, 676, 790, 991]
    plots = [show_prediction(model, tile) for tile in tiles]
    @pipe showimg(plots, (5000, 7000), (7, 1)) |> savefig(_, filename)
    
end

function mainFunction()
    println(CUDA.functional())
    train_data, test_data = load_data(0.8)
    model = get_model()
    train_model(model, train_data)
end

main = mainFunction()
