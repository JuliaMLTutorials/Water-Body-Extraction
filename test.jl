using Plots, Images, ArchGDAL, Flux, Augmentor, MLUtils, UNet, CUDA, Random, ProgressBars
using Pipe: @pipe

function read_rgb(tile::Int)::Array{Float32,3}
	img = ArchGDAL.readraster("data/rgb/rgb.$tile.tif")[:,:,:]
	scale = Float32(findmax(img)[1])
	@pipe img .|> Float32 .|> /(_, scale)
end;

function read_nir(tile::Int)::Array{Float32,3}
	img = ArchGDAL.readraster("data/nir/nir.$tile.tif")[:,:,:]
	thresholded_img = @pipe img .|> Float32 |> clamp.(_, Float32(0), Float32(3000))
    scale = findmax(thresholded_img)[1] |> Float32
	thresholded_img ./ scale
end;

function read_swir(tile::Int)::Array{Float32,3}
	img = ArchGDAL.readraster("data/swir/swir.$tile.tif")[:,:,:]
	thresholded_img = @pipe img .|> Float32 |> clamp.(_, Float32(0), Float32(3000))
	thresholded_img ./ Float32(findmax(thresholded_img)[1])
end;

function read_mask(tile::Int)::Array{Float32,3}
	@pipe ArchGDAL.readraster("data/mask/mask.$tile.tif")[:,:,:] .|> 
	/(_, UInt8(255)) .|> 
	Float32
end;

function onehot_mask(mask::Array{Float32,4}, nclasses::Int)
	rows, cols, _, obs = size(mask)
	mask_out = zeros(Float32, (rows, cols, nclasses, obs))
	for (r, c, o) in [(r, c, o) for r in 1:rows for c in 1:cols for o in 1:obs]
		onehot_index = mask[r,c,1,o] + 1 |> floor |> Int
		mask_out[r,c,onehot_index,o] = 1.0
	end
	return mask_out
end;

"A struct representing a pipeline for a selection of patches."
struct ImagePipeline
    tiles::Vector{Int}
end

function Base.length(X::ImagePipeline)
	return length(X.tiles)
end

function Base.getindex(X::ImagePipeline, i::Union{<:AbstractArray,Int})
    i = i isa AbstractArray ? i : [i]
	tiles = X.tiles[i]
	xs = zeros(Float32, (512, 512, 5, length(i)))
	ys = zeros(Float32, (512, 512, 1, length(i)))
	@Threads.threads for (obs, tile) in collect(enumerate(tiles))
		rgb = @pipe read_rgb(tile)
		nir = @pipe read_nir(tile)
		swir = @pipe read_swir(tile)
		mask = @pipe read_mask(tile)
		xs[:,:,1:3,obs] .= rgb
		xs[:,:,4,obs] .= nir
		xs[:,:,5,obs] .= swir
		ys[:,:,1,obs] .= mask
	end
    return xs |> gpu, onehot_mask(ys, 2) |> gpu
end

function dice_loss(ŷ::AbstractArray{Float32, 4}, y::AbstractArray{Float32, 4})
    intersection = sum(ŷ .* y, dims=(1, 2, 4))
    union = sum(ŷ, dims=(1, 2, 4)) .+ sum(y, dims=(1, 2, 4))
    dice_coefficient = ((2.0f0 .* intersection) .+ eps(Float32)) ./ (union .+ eps(Float32))
    return mean(1.0f0 .- dice_coefficient)
end

function get_model()
    Chain(Unet(5, 2), x -> softmax(x, dims=3)) |> gpu
end

function train_model(model)
	# Load Data
    tiles = [i for i in 600:1600]
    shuffle!(tiles)
	data = ImagePipeline(tiles)
	train_data = DataLoader(data, batchsize=1)
	
	# Define Loss Function
	loss(x, y) = Flux.crossentropy(model(x), y, dims=3)
    """
	function loss(x, y) 
        ŷ = model(x)
        return (0.5f0 * dice_loss(ŷ[:,:,2:2,:], y[:,:,2:2,:])) + (0.5f0 * Flux.crossentropy(ŷ, y, dims=3))
    end
    """

	# Define Optimizer
	opt = Flux.Optimise.ADAM(1e-4)

	# Get Parameters
	params = Flux.params(model)

    # Train For Two Epoch
	for epoch in 1:2

        i = 0
        l = 0.0f0
        iter = ProgressBar(train_data)
		for (x, y) in iter

            water_content = @pipe sum(y[:,:,2,:]) |> /(_, 512 * 512) |> *(_, 100)
            # println(water_content)

            if water_content > 0.0

                # Compute Gradients
                grads = Flux.gradient(() -> loss(x, y), params)

                # Update Parameters
                Flux.Optimise.update!(opt, params, grads)

                i += 1
                l += loss(x, y)
                set_description(iter, "Average Loss: $(l / Float32(i))")
		
            end
		end
	end
end