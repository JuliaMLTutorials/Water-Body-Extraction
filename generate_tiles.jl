module GenerateTiles
    
using ArchGDAL, Images, Plots
using Pipe: @pipe

function write_raster(filename, buffer)
    (width, height, nbands) = size(buffer)
    raster_to_write = ArchGDAL.create(filename, width=width, height=height, nbands=nbands, dtype=eltype(buffer), driver=ArchGDAL.getdriver("GTiff"))
	for band in 1:nbands
		ArchGDAL.write!(raster_to_write, buffer[:,:,band], band)
	end
	ArchGDAL.destroy(raster_to_write)
end

function make_tiles()
    # Load RGB and NIR
    rgb_nir = ArchGDAL.readraster("data/init/rgb_nir.tif")
    width, height = ArchGDAL.width(rgb_nir), ArchGDAL.height(rgb_nir)
    upper_bound = min(ArchGDAL.width(rgb_nir), ArchGDAL.height(rgb_nir)) - 1023
    rgb = rgb_nir[:,:,1:3]
    nir = rgb_nir[:,:,4:4]
    @debug "Type Of RGB: $(typeof(rgb)), Size Of RGB: $(size(rgb))"
    @debug "Type Of NIR: $(typeof(nir)), Size Of NIR: $(size(nir))"

    # Load SWIR
    swir = @pipe ArchGDAL.readraster("data/init/swir.tif")[:,:,1] |> imresize(_, Int(width), Int(height)) |> reshape(_, (size(_)..., 1)) |> floor.(_) |>  UInt16.(_)
    @debug "Type Of SWIR: $(typeof(swir)), Size Of SWIR: $(size(swir))"

    # Load Mask
    label = ArchGDAL.readraster("data/init/label.tif")[:,:,:]
    @debug "Type Of Mask: $(typeof(label))), Size Of Mask: $(size(label))"
    tile = 100

    for (tile, (row, col)) in enumerate((row, col) for row in 1:1024:upper_bound for col in 1:1024:upper_bound)
        write_raster("data/rgb/rgb.$tile.tif", rgb[col:col+1023,row:row+1023,:])
        write_raster("data/nir/nir.$tile.tif", nir[col:col+1023,row:row+1023,:])
        write_raster("data/swir/swir.$tile.tif", swir[col:col+1023,row:row+1023,:])
        write_raster("data/mask/mask.$tile.tif", label[col:col+1023,row:row+1023,:])
        @debug "Wrote Tile: $tile"
   end
end

main = make_tiles()

end