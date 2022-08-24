using Statistics, ArchGDAL, Flux
using Pipe: @pipe

function read_rgb(tile::Int)::Array{Float32,3}
	ArchGDAL.readraster("data/rgb/rgb.$tile.tif")[:,:,:]
end;

function read_nir(tile::Int)::Array{Float32,3}
	ArchGDAL.readraster("data/nir/nir.$tile.tif")[:,:,:]
end;

function read_swir(tile::Int)::Array{Float32,3}
	ArchGDAL.readraster("data/swir/swir.$tile.tif")[:,:,:]
end;

function read_mask(tile::Int)::Array{Float32,3}
	@pipe ArchGDAL.readraster("data/mask/mask.$tile.tif")[:,:,:] .|> Float32 .|> /(_, 255.0f0)
end;

function onehot_mask(mask::Array{Float32,4}, nclasses::Int)
	rows, cols, _, obs = size(mask)
	mask_out = zeros(Float32, (rows, cols, nclasses, obs))
	for (r, c, o) in [(r, c, o) for r in 1:rows for c in 1:cols for o in 1:obs]
		onehot_index = mask[r,c,1,o] + 1 |> floor |> Int
		mask_out[r,c,onehot_index,o] = 1.0f0
	end
	return mask_out
end;

function normalize_rgb(img::Array{Float32,3})
	μ = mean(img, dims=(1, 2))
	σ = std(img, dims=(1, 2))
	@pipe (img .- μ) ./ σ |> reshape(_, size(img)..., 1)
end

function normalize_gray(img::Array{Float32,3})
	thresholded_img = clamp.(img, 0.0f0, 3000.0f0)
	normalize_rgb(thresholded_img)
end

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
		rgb = read_rgb(tile) |> normalize_rgb
		nir = read_nir(tile) |> normalize_gray
		swir = read_swir(tile) |> normalize_gray
		mask = read_mask(tile)
		xs[:,:,1:3,obs] .= rgb
		xs[:,:,4,obs] .= nir
		xs[:,:,5,obs] .= swir
		ys[:,:,1,obs] .= mask
	end
    return xs[:,:,1:5,:] |> gpu, onehot_mask(ys, 2) |> gpu
end
