module UNetModule

using Flux 
using Pipe: @pipe

function ConvBlock(kernel, in, out)
    Chain(
        Conv((kernel, kernel), in=>out, pad=SamePad()), 
        BatchNorm(out), 
        x -> relu.(x)
    )
end

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

struct UNet
    encoder_blocks
    decoder_blocks
    activation
end

Flux.@functor UNet

function UNet(channels::Int, nclasses::Int, filters=[32, 64, 128, 256, 512])
    @assert length(filters) == 5
    encoder_blocks = vcat([EncoderBlock(channels, filters[1])], [EncoderBlock(filters[i], filters[i+1]) for i in 1:4])
    decoder_blocks = [DecoderBlock(filters[i]+filters[i+1], filters[i]) for i in 1:4]
    activation = Chain(Conv((3, 3), filters[1]=>nclasses, pad=SamePad()), x -> softmax(x, dims=3))
    UNet(encoder_blocks, decoder_blocks, activation)
end

function (l::UNet)(x)
    x1, skip1 = l.encoder_blocks[1](x)
    x2, skip2 = l.encoder_blocks[2](x1)
    x3, skip3 = l.encoder_blocks[3](x2)
    x4, skip4 = l.encoder_blocks[4](x3)
    _, out = l.encoder_blocks[5](x4)

    up1 = l.decoder_blocks[4](out, skip4)
    up2 = l.decoder_blocks[3](up1, skip3)
    up3 = l.decoder_blocks[2](up2, skip2)
    up4 = l.decoder_blocks[1](up3, skip1)

    return l.activation(up4)
end

end