### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 1bbaf788-859e-4cff-bc0e-3beaa0bc70e1
# ╠═╡ show_logs = false
begin
	using Pkg
	Pkg.activate(".")
	using Plots, Images, ArchGDAL, Flux, MLUtils, Statistics
	using Pipe: @pipe
end;

# ╔═╡ c291f91f-2d0b-4a1d-bd5b-f83afa915107
begin
	using CUDA
	CUDA.functional() |> println
end

# ╔═╡ 9cebd545-7d2a-42c8-b5c3-e466891175af
using Random: shuffle!

# ╔═╡ ab676064-2c28-4dda-9a46-b8fb68a132cc
begin
	using BSON: @load
	@load "model.bson" model
	model = model |> gpu
end;

# ╔═╡ c26ac38f-02be-42d4-a2cd-5b0fa8e50cfe
function read_rgb(tile::Int)::Array{Float32,3}
	ArchGDAL.readraster("data/rgb/rgb.$tile.tif")[:,:,:]
end;

# ╔═╡ 9564049a-3afa-4340-9d26-7949d441fe12
function read_nir(tile::Int)::Array{Float32,3}
	ArchGDAL.readraster("data/nir/nir.$tile.tif")[:,:,:]
end;

# ╔═╡ f60858fd-f979-4e9d-96be-c31aaa600512
function read_swir(tile::Int)::Array{Float32,3}
	ArchGDAL.readraster("data/swir/swir.$tile.tif")[:,:,:]
end;

# ╔═╡ 56396a5b-aa4e-4165-97ae-be0ad13b07ac
function read_mask(tile::Int)::Array{Float32,3}
	@pipe ArchGDAL.readraster("data/mask/mask.$tile.tif")[:,:,:] .|> 
	Float32 .|> 
	/(_, 255.0f0)
end;

# ╔═╡ 309aba63-516f-4726-8509-f51f3ed4f00a
begin
	function showimg(img::AbstractVector, layout::Tuple{Int,Int}, sz::Tuple{Int,Int})
		plot(img..., layout=layout, size=sz, axis=nothing, showaxis=false, margin=0Plots.mm)
	end

	function showimg(img::Any, sz::Tuple{Int,Int})
		plot(img, size=sz, axis=nothing, showaxis=false, margin=0Plots.mm)
	end

	function put_title!(plot, title::String)
		Plots.title!(plot, title, titlefontsize=45)
	end
end;

# ╔═╡ 598fdf7d-5de9-4918-b885-ccdd5c2959e0
function plot_color(img::Array{Float32,3}, gamma=1.0)
	scale = @pipe findmax(img)[1] |> Float32 |> max(_, 1.0f0)
	@pipe img .|> 
	/(_, scale) |>
	permutedims(_, (3, 2, 1)) |> 
	colorview(RGB, _) |> 
	adjust_histogram(_, GammaCorrection(gamma=gamma)) |>
	showimg(_, (500, 500))
end;

# ╔═╡ 90ab5f98-91d9-4b6e-8499-83e00fd3568a
function plot_gray(img::Array{Float32,3})
	scale = @pipe findmax(img)[1] |> Float32 |> max(_, 1.0f0)
	@pipe img[:,:,1] |> 
	/(_, scale) |>
	permutedims(_, (2, 1)) |> 
	colorview(Gray, _) |>
	showimg(_, (500, 500))
end;

# ╔═╡ 13c5a2a8-0976-4a11-a7de-707f4d3ec13a
function plot_sample(tile::Int)
	rgb_plot = @pipe read_rgb(tile) |> plot_color(_, 0.8) |> put_title!(_, "RGB")
	nir_plot = @pipe read_nir(tile) |> plot_gray(_) |> put_title!(_, "NIR")
	swir_plot = @pipe read_swir(tile) |> plot_gray(_) |> put_title!(_, "SWIR")
	mask_plot = @pipe read_mask(tile) |> plot_gray(_) |> put_title!(_, "Mask")
	plots = [rgb_plot, nir_plot, swir_plot, mask_plot]
	showimg(plots, (1, 4), (2000, 600))
end;

# ╔═╡ 0025f82d-9b69-4761-8fca-7521a788c448
function plot_samples(nsamples)
	samples = rand(1:nsamples, 3)
	samples = [1353, 1274, 520, 1571]
	plots = [plot_sample(sample) for sample in samples]
	showimg(plots, (4, 1), (2000, 2300))
end;

# ╔═╡ 84a678d9-890e-41b4-9076-c391da7f0660
plot_samples(1600)

# ╔═╡ 2a788793-86cf-49c7-98b0-a8c16e536910
function onehot_mask(mask::Array{Float32,4}, nclasses::Int)
	rows, cols, _, obs = size(mask)
	mask_out = zeros(Float32, (rows, cols, nclasses, obs))
	for (r, c, o) in [(r, c, o) for r in 1:rows for c in 1:cols for o in 1:obs]
		onehot_index = mask[r,c,1,o] + 1 |> floor |> Int
		mask_out[r,c,onehot_index,o] = 1.0f0
	end
	return mask_out
end;

# ╔═╡ 95e427b2-bcc5-4010-866b-59e48cec0a95
function normalize_rgb(img::Array{Float32,3})
	μ = mean(img, dims=(1, 2))
	σ = std(img, dims=(1, 2))
	@pipe (img .- μ) ./ σ |> reshape(_, size(img)..., 1)
end;

# ╔═╡ 11cbe89a-2895-4f82-af90-7b5666b84c07
function normalize_gray(img::Array{Float32,3})
	thresholded_img = clamp.(img, 0.0f0, 3000.0f0)
	normalize_rgb(thresholded_img)
end;

# ╔═╡ 439cad39-729f-4342-bfaf-4fe23ec860f4
struct ImagePipeline
    tiles::Vector{Int}
end;

# ╔═╡ 5c46e24e-70ee-4dd7-acf2-0276a1e35c90
function Base.length(X::ImagePipeline)
	return length(X.tiles)
end;

# ╔═╡ adaf4997-6522-4b09-8fa7-da541c402856
function Base.getindex(X::ImagePipeline, i::Union{<:AbstractArray,Int})
    i = i isa AbstractArray ? i : [i]
	tiles = X.tiles[i]
	xs = zeros(Float32, (512, 512, 5, length(i)))
	ys = zeros(Float32, (512, 512, 1, length(i)))
	@Threads.threads for (obs, tile) in collect(enumerate(tiles))
		rgb = read_rgb(tile) |> normalize_rgb
		nir = read_nir(tile) |> normalize_gray
		swir = read_swir(tile) |> normalize_gray
		mask = read_mask(tile)
		xs[:,:,1:3,obs] .= rgb
		xs[:,:,4,obs] .= nir
		xs[:,:,5,obs] .= swir
		ys[:,:,1,obs] .= mask
	end
    return xs |> gpu, onehot_mask(ys, 2) |> gpu
end

# ╔═╡ ba6704fe-ecde-44c7-ade9-88a4ab17bd56
md"""
# Activate The Environment
For the sake of reproducibility, we need to activate the environment specified by the provided `Project.toml` file. If you have not already done so, you should make sure to instantiate this environment before attempting the run this notebook.
"""

# ╔═╡ 27caf607-df1e-45a1-a4e8-2d5394de5b53
md"""
# Confirm The GPU Is Available
The training of segmentation models is generally resource intensive, and thus benefits greatly from the presence of GPU acceleration. To confirm that the GPU is available, we simply import the `CUDA` module and call `CUDA.functional()`. This method will return `true` if GPU acceleration is available and `false` otherwise.
"""

# ╔═╡ d4d3b5b4-6374-43f2-b53c-c664ba8e0b8b
md"""
# Working With Raster Data
To work with remotely sensed imagery, we typically need to be able to read and write to raster files. Julia provides the [ArchGDAL](https://yeesian.com/ArchGDAL.jl/latest/) package for this purpose, which exposes an API similar to that of the popular Python library [rasterio](https://rasterio.readthedocs.io/en/latest/). Here we demonstrate how to read the RGB, NIR, and SWIR features from disk along with the corresponding mask.
"""

# ╔═╡ b62a9f84-1cce-11ed-1787-d3d0a9957adf
md"""
# Visualize The Data
It's generally much easier to understand the problem domain once we are able to actually visualize the data we're working with. This is especially true for computer vision, as it allows us to easily identify any potential strengths or weaknesses inherent to the dataset. The data we're using for this example was initially sourced from [here](https://github.com/SCoulY/Sentinel-2-Water-Segmentation). However, I have [provided](https://drive.google.com/file/d/1Q8lzwO1kJHd3BCdY6aajhxkDyhMhDA1T/view?usp=sharing) a simplified and pre-processed version for the sake of this demonstration. 

The dataset consists of 1600 512x512 samples which were generated by slicing the initial 20982x20982 image into non-overlapping tiles. Each sample provides us with the RGB, NIR, and SWIR bands, taken from Sentinel-2 satellite imagery collected over Chengdu City in Sichuan Province, China. Each sample is paired with a labelled mask, which we use to provide the ground-truth when training our model. The code below plots the RGB, NIR, and SWIR features for four assorted samples along with their corresponding masks.
"""

# ╔═╡ 1b5a025f-4900-4977-8f7d-72bd5679e932
md"""
# Create The Data Pipeline
Our data pipeline is responsible for reading both the input features and mask from disk, applying data normalization, and finally converting the images into tensors compatible with Flux's Convolutional layers. Due to the size of the dataset, we will lazily read images from disk as they are needed rather than front-loading everything into memory.

The `ImagePipeline` struct, which is defined below, will be used to leverage Julia's multiple-dispatch system in order to make Flux's machinery call upon our own methods during training and inference. Later, we will pass our `ImagePipeline` struct into `DataLoader(data)`, which expects that data implements the `Base.length()` and `Base.get_index()` methods. Once we have defined these methods for our custom data type, it will be compatible the the rest of the Flux ecosystem, allowing us to define our own behaviour for the data pipeline.
"""

# ╔═╡ 445df035-1888-4119-a94c-866a9f4e7ff4
md"""
# Define The Model
[U-Net](https://arxiv.org/abs/1505.04597) is a popular and widely used model in the field of image segmentation. Its architecture consists of a symmetric arrangement of encoder and decoder blocks, which are connected by skip connections to mix spatially-rich features from the encoder with semantically-rich features in the decoder. Here we demonstrate how to define our own U-Net from scratch with Flux. In general, Flux provides us with two distinct approaches for defining our own layers. The first and by far the most common approach involves composing one or more of Flux's pre-defined layers with a higher-order layer such as `Chain`, `SkipConnection`, or `Parallel`. The second approach is more involved, but gives us finer control over our layer's behaviour, and generally consists of four distinct steps:
1. Define a struct to store our layer's internal state.
2. Call `Flux.@functor` on our struct to make it compatible with Flux's machinery.
3. Define a constructor to initialize our layer.
4. Define the forward-pass by making the struct callable on the intended input.

It should be noted that Flux does not make a distinction between models and layers, as layers can themselves be defined as arbitrarily complex compositions of other layers.
"""

# ╔═╡ d2428330-8e46-4547-b867-7e2a656357b7
function ConvBlock(kernel, in, out)
    Chain(
        Conv((kernel, kernel), in=>out, pad=SamePad()), 
        BatchNorm(out), 
        x -> relu.(x)
    )
end;

# ╔═╡ 3e013263-2736-41e9-825b-8f5177baa451
begin
	struct EncoderBlock
    	conv_block
    	downsample
	end

	Flux.@functor EncoderBlock

	function EncoderBlock(in::Int, out::Int)
    	conv_block = Chain(ConvBlock(3, in, out), ConvBlock(3, out, out))
    	downsample = MaxPool((2, 2), pad=SamePad())
    	EncoderBlock(conv_block, downsample)
	end

	function (l::EncoderBlock)(x)
    	skip = l.conv_block(x)
    	return l.downsample(skip), skip
	end
end;

# ╔═╡ 3858d2ba-5738-4551-9d81-5ddc459af114
begin
	struct DecoderBlock
    	conv_block
    	upsample
	end

	Flux.@functor DecoderBlock

	function DecoderBlock(in::Int, out::Int)
		conv_block = Chain(ConvBlock(3, in, out), ConvBlock(3, out, out))
		upsample = Upsample(:bilinear, scale=(2, 2))
		DecoderBlock(conv_block, upsample)
	end

	function (l::DecoderBlock)(x, skip)
		return @pipe l.upsample(x) |> cat(_, skip, dims=3) |> l.conv_block
	end
end;

# ╔═╡ c2a0e450-ec5e-45dd-9d31-6ac64398a868
begin
	struct UNet
    	encoder_blocks
    	decoder_blocks
    	activation
	end

	Flux.@functor UNet

	function UNet(channels::Int, nclasses::Int, filters=[32, 64, 128, 256, 512])
		# Build Encoder
		@assert length(filters) == 5
		input_block = [EncoderBlock(channels, filters[1])]
		remaining_blocks = [EncoderBlock(filters[i], filters[i+1]) for i in 1:4]
		encoder_blocks = vcat(input_block, remaining_blocks)

		# Build Decoder
		decoder_filters = [(filters[i]+filters[i+1], filters[i]) for i in 1:4]
    	decoder_blocks = [DecoderBlock(f[1], f[2]) for f in decoder_filters]
    	
		# Build Activation Layer
		activation = Chain(
			Conv((3, 3), filters[1]=>nclasses, pad=SamePad()), 
			x -> softmax(x, dims=3)
		)

		# Assemble UNet
    	UNet(encoder_blocks, decoder_blocks, activation)
	end

	function (l::UNet)(x)
    	# Forward Pass Through Encoder
		x1, skip1 = l.encoder_blocks[1](x)
    	x2, skip2 = l.encoder_blocks[2](x1)
    	x3, skip3 = l.encoder_blocks[3](x2)
    	x4, skip4 = l.encoder_blocks[4](x3)
    	_, out = l.encoder_blocks[5](x4)

    	# Forward Pass Through Decoder
		up1 = l.decoder_blocks[4](out, skip4)
    	up2 = l.decoder_blocks[3](up1, skip3)
    	up3 = l.decoder_blocks[2](up2, skip2)
    	up4 = l.decoder_blocks[1](up3, skip1)

    	# Run Final Classification Layer
		return l.activation(up4)
	end
end;

# ╔═╡ 585a36a1-b878-470f-9710-edd70f5c1559
md"""
# Define The Loss
Our dataset is highly imbalanced with respect to water, with about 2.18% of all pixels in the ground-truth being labelled as such. This issue is common in the field of image segmentation, as regions of interest typically occupy only a small region of the field of view. While pixelwise cross entropy remains a popular choice, the Dice loss has demonstrated good performance on such tasks, particularly when facing extreme class imbalance.
"""

# ╔═╡ ae9d1224-a9a2-4d2f-a2df-0bc99839b4cd
function dice_loss(ŷ::AbstractArray{Float32, 4}, y::AbstractArray{Float32, 4})
	ϵ = eps(Float32)
    intersection = sum(ŷ .* y, dims=(1, 2, 4))
    union = sum(ŷ, dims=(1, 2, 4)) .+ sum(y, dims=(1, 2, 4))
    dice_coefficient = ((2.0f0 .* intersection) .+ ϵ) ./ (union .+ ϵ)
    return mean(1.0f0 .- dice_coefficient)
end;

# ╔═╡ 1172e97b-9837-4074-a622-4dddbf74e1ce
md"""
# Train The Model

Armed with our data pipeline, model, and loss, we're finally ready to begin training our model. While Flux provides us with the convenience function `train!(loss, params, data, opt)` for easy training, it is often advantageous to define our own training loop for finer control. While we provide the training code here for educational purposes, we will actually be loading a pre-trained model for the sake of practicality. This is due to the high memory requirements typically placed on the training of segmentation models, with a minimum of 6GB of VRAM required in order to run this example. Should you wish, you may replace the code for loading the pre-trained model with a call to `train_model()` to run the training algorithm for yourself.
"""

# ╔═╡ 86144ce0-a288-43f0-a0a2-ae7a5948c993
const SAMPLE_TILES = [114, 236, 318, 669, 676, 790, 991];

# ╔═╡ 5171cbc0-192b-48ae-a427-d26e53e26520
function load_dataset()
	# List All Tiles
    tiles = collect(1:1600)
    filter!(x -> !(x in SAMPLE_TILES), tiles)
    shuffle!(tiles)

	# Split Tiles Into Training And Test
	split_index = 1600 * 0.8 |> floor |> Int
	train = ImagePipeline(tiles[1:split_index])
	test = ImagePipeline(tiles[split_index+1:end])

	# Construct DataLoaders
	train_data = DataLoader(train, batchsize=2, shuffle=true)
	test_data = DataLoader(test, batchsize=15, shuffle=false)
	return train_data, test_data
end;

# ╔═╡ a9b0bddc-954d-48c5-8fbb-0bffe1b2efba
function train_model(model, train_data)
	# Define Optimizer
	opt = Flux.Optimiser(ClipNorm(1e-4), Adam(1e-4))

	# Define Loss
	loss(x, y) = dice_loss(model(x), y)

	# Get Parameters
	params = Flux.params(model)

	# Train For Two Epochs
	for epoch in 1:2

        # Iterate Over Data
		total_loss = 0.0f0
		for (i, (x, y)) in enumerate(train_data)

            # Compute Gradient
            grads = Flux.gradient(() -> loss(x, y), params)

            # Update Parameters
            Flux.Optimise.update!(opt, params, grads)

            # Log Progress    
			current_loss = loss(x, y)
            total_loss += current_loss
			println("Loss: $current_loss, Average Loss: $(total_loss / Float32(i))")
        end
	end
end;

# ╔═╡ d9048f8a-2a74-4c7c-a106-004c7e619e9b
train_data, test_data = load_dataset();

# ╔═╡ 3fd7e85a-529e-4c21-8ac1-5ce4cbe9e2c1
# ╠═╡ show_logs = false
begin
	Core.eval(Main, :(include("UNet.jl")))
	Core.eval(Main, :(using CUDA))
	Core.eval(Main, :(using Flux))
end;

# ╔═╡ 23bea54f-5608-4bfb-b62e-79fa7349a8b0
function prediction_to_onehot(ŷ::Array{Float32,4})
    rows, cols, _, obs = size(ŷ)
    onehot = zeros(Float32, size(ŷ))
    for row in 1:rows, col in 1:cols, ob in 1:obs
        index = argmax(ŷ[row,col,:,ob])
        onehot[row,col,index,ob] = 1.0f0
    end
    return onehot
end;

# ╔═╡ 10e32b58-bdca-4ea9-a106-d11529eaf5e8
function true_positives(ŷ::Array{Float32,4}, y::Array{Float32,4})
    return @pipe ŷ .* y |> sum(_, dims=(1, 2, 4)) |> reshape(_, size(y)[3])
end;

# ╔═╡ c26f6963-05a7-4fa6-a6d6-978b48784c08
function false_positives(ŷ::Array{Float32,4}, y::Array{Float32,4})
    return @pipe ŷ .* (1 .- y) |> sum(_, dims=(1, 2, 4)) |> reshape(_, size(y)[3])
end;

# ╔═╡ ae774bdc-a62b-4bd5-a238-69b3c6e190f0
function false_negatives(ŷ::Array{Float32,4}, y::Array{Float32,4})
    return @pipe (1 .- ŷ) .* y |> sum(_, dims=(1, 2, 4)) |> reshape(_, size(y)[3])
end;

# ╔═╡ 023ae564-36b3-4b2d-9ede-c2feca60c672
function precision(ŷ::Array{Float32,4}, y::Array{Float32,4})
    ŷ = prediction_to_onehot(ŷ)
   	tp = true_positives(ŷ, y)
    fp = false_positives(ŷ, y)
    return tp ./ (tp .+ fp)
end;

# ╔═╡ d05be475-5bf4-4604-8746-f44bd80f861e
function recall(ŷ::Array{Float32,4}, y::Array{Float32,4})
    ŷ = prediction_to_onehot(ŷ)
    tp = true_positives(ŷ, y)
    fn = false_negatives(ŷ, y)
    return tp ./ (tp .+ fn)
end;

# ╔═╡ be6b5cfc-4eb5-43f7-8b32-4af2ab41c526
function IoU(ŷ::Array{Float32,4}, y::Array{Float32,4})
    ϵ = eps(Float32)
    ŷ = prediction_to_onehot(ŷ)
    tp = true_positives(ŷ, y)
    fn = false_negatives(ŷ, y)
    fp = false_positives(ŷ, y)
    return (tp .+ ϵ) ./ (tp .+ fp .+ fn .+ ϵ)
end;

# ╔═╡ 07151940-e03a-493d-965d-40a6da92dd22
function mIoU(ŷ::Array{Float32,4}, y::Array{Float32,4})
    return IoU(ŷ, y) |> mean
end;

# ╔═╡ 0677f844-e0cb-45fc-98c2-e977ee66d25f
# ╠═╡ show_logs = false
begin
	total_mIoU = 0.0
	total_recall = [0.0, 0.0]
	total_precision = [0.0, 0.0]
	for (x, y) in test_data
		ŷ, y = Array(model(x)), Array(y)
		total_mIoU += mIoU(ŷ, y)
		total_recall .+= recall(ŷ, y)
		total_precision .+= precision(ŷ, y)
	end
	average_mIoU = total_mIoU / length(test_data)
	average_recall = total_recall ./ length(test_data)
	average_precision = total_precision ./ length(test_data)
end;

# ╔═╡ 6c1bb1ff-d277-4f5a-96e9-a3424e7a00f0
md"""
# Quantitative Analysis
Now that we've trained our model, we need to quantify our final performance on the test data. For this purpose, we have defined methods for calculating the precision, recall, and mIoU of our model. When analyzing the results, we immediately observe that our model is much better at detecting background than water. This is due to the relative scarcity of water in comparison to background in our dataset, which causes our loss function to prioritize accurate labelling of the latter over the former. A more sophisticated approach may be employed to address this problem, but that is beyond the scope of this tutorial.

## Results:
##### mIoU: $(round(average_mIoU * 100.0, digits=4, base=10))
##### Recall (Background): $(round(average_recall[1] * 100.0, digits=4, base=10))
##### Recall (Water): $(round(average_recall[2] * 100.0, digits=4, base=10))
##### Precision (Background): $(round(average_precision[1] * 100.0, digits=4, base=10))
##### Precision (Water): $(round(average_precision[2] * 100.0, digits=4, base=10))

"""

# ╔═╡ 63720864-8a28-4fd2-9750-f79ad6214e3f
md"""
# Qualitative Analysis
While quantitative metrics are all well and good, it's a good idea to perform a qualitative analysis as well. Here we plot the features, masks, and predictions for the 7 sample tiles which we previously removed from our dataset. Because our model has not seen these samples during training, they can be used to provide a reasonable estimate of its real-world performance.
"""

# ╔═╡ 0b5dd085-8456-4d16-b92b-7a098b259235
function prediction_to_mask(ŷ::Array{Float32,4})
	(mapslices(argmax, ŷ, dims=3) .|> Float32) .- 1.0f0
end;

# ╔═╡ 83aef662-41b6-40b2-b952-1c195255ad58
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
	showimg([rgb_plot, nir_plot, swir_plot, mask_plot, prediction_plot], (1, 5), (2500, 1000))
end;

# ╔═╡ e2307d85-2b0b-46a8-9ffd-1e542930f4a4
function perform_qualitative_analysis()
	plots = [show_prediction(model, tile) for tile in SAMPLE_TILES]
	showimg(plots, (7, 1), (2500, 3500))
end;

# ╔═╡ e71f9907-59ff-4f0b-a959-649be08a224e
# ╠═╡ show_logs = false
perform_qualitative_analysis()

# ╔═╡ Cell order:
# ╟─ba6704fe-ecde-44c7-ade9-88a4ab17bd56
# ╠═1bbaf788-859e-4cff-bc0e-3beaa0bc70e1
# ╟─27caf607-df1e-45a1-a4e8-2d5394de5b53
# ╠═c291f91f-2d0b-4a1d-bd5b-f83afa915107
# ╟─d4d3b5b4-6374-43f2-b53c-c664ba8e0b8b
# ╠═c26ac38f-02be-42d4-a2cd-5b0fa8e50cfe
# ╠═9564049a-3afa-4340-9d26-7949d441fe12
# ╠═f60858fd-f979-4e9d-96be-c31aaa600512
# ╠═56396a5b-aa4e-4165-97ae-be0ad13b07ac
# ╟─b62a9f84-1cce-11ed-1787-d3d0a9957adf
# ╠═309aba63-516f-4726-8509-f51f3ed4f00a
# ╠═598fdf7d-5de9-4918-b885-ccdd5c2959e0
# ╠═90ab5f98-91d9-4b6e-8499-83e00fd3568a
# ╠═13c5a2a8-0976-4a11-a7de-707f4d3ec13a
# ╠═0025f82d-9b69-4761-8fca-7521a788c448
# ╟─84a678d9-890e-41b4-9076-c391da7f0660
# ╟─1b5a025f-4900-4977-8f7d-72bd5679e932
# ╠═2a788793-86cf-49c7-98b0-a8c16e536910
# ╠═95e427b2-bcc5-4010-866b-59e48cec0a95
# ╠═11cbe89a-2895-4f82-af90-7b5666b84c07
# ╠═439cad39-729f-4342-bfaf-4fe23ec860f4
# ╠═5c46e24e-70ee-4dd7-acf2-0276a1e35c90
# ╠═adaf4997-6522-4b09-8fa7-da541c402856
# ╟─445df035-1888-4119-a94c-866a9f4e7ff4
# ╠═d2428330-8e46-4547-b867-7e2a656357b7
# ╠═3e013263-2736-41e9-825b-8f5177baa451
# ╠═3858d2ba-5738-4551-9d81-5ddc459af114
# ╠═c2a0e450-ec5e-45dd-9d31-6ac64398a868
# ╟─585a36a1-b878-470f-9710-edd70f5c1559
# ╠═ae9d1224-a9a2-4d2f-a2df-0bc99839b4cd
# ╟─1172e97b-9837-4074-a622-4dddbf74e1ce
# ╠═9cebd545-7d2a-42c8-b5c3-e466891175af
# ╠═86144ce0-a288-43f0-a0a2-ae7a5948c993
# ╠═5171cbc0-192b-48ae-a427-d26e53e26520
# ╠═a9b0bddc-954d-48c5-8fbb-0bffe1b2efba
# ╠═d9048f8a-2a74-4c7c-a106-004c7e619e9b
# ╠═3fd7e85a-529e-4c21-8ac1-5ce4cbe9e2c1
# ╠═ab676064-2c28-4dda-9a46-b8fb68a132cc
# ╟─6c1bb1ff-d277-4f5a-96e9-a3424e7a00f0
# ╠═23bea54f-5608-4bfb-b62e-79fa7349a8b0
# ╠═10e32b58-bdca-4ea9-a106-d11529eaf5e8
# ╠═c26f6963-05a7-4fa6-a6d6-978b48784c08
# ╠═ae774bdc-a62b-4bd5-a238-69b3c6e190f0
# ╠═023ae564-36b3-4b2d-9ede-c2feca60c672
# ╠═d05be475-5bf4-4604-8746-f44bd80f861e
# ╠═be6b5cfc-4eb5-43f7-8b32-4af2ab41c526
# ╠═07151940-e03a-493d-965d-40a6da92dd22
# ╠═0677f844-e0cb-45fc-98c2-e977ee66d25f
# ╟─63720864-8a28-4fd2-9750-f79ad6214e3f
# ╠═0b5dd085-8456-4d16-b92b-7a098b259235
# ╠═83aef662-41b6-40b2-b952-1c195255ad58
# ╠═e2307d85-2b0b-46a8-9ffd-1e542930f4a4
# ╟─e71f9907-59ff-4f0b-a959-649be08a224e
