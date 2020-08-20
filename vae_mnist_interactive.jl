### A Pluto.jl notebook ###
# v0.11.2

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 1338fbd0-d62f-11ea-24b1-0fc6f3dec14b
using Pkg

# ╔═╡ 819ebe60-d630-11ea-3c01-0b746169f32b
begin
	using Base.Iterators: partition
	using Flux
	using BSON
	using CUDAapi: has_cuda_gpu
	using DrWatson: struct2dict
	using Flux: logitbinarycrossentropy, chunk
	using Flux.Data: DataLoader
	using Images
	using Logging: with_logger
	using MLDatasets
	using Parameters: @with_kw
	using ProgressMeter: Progress, next!
	using TensorBoardLogger: TBLogger, tb_overwrite
	using Random
	using CuArrays, CUDAdrv
	using Plots
	using PlutoUI
end

# ╔═╡ 91e51d60-d62f-11ea-32c5-490645dfe7fe
Pkg.activate(".")

# ╔═╡ a42e4940-e2bb-11ea-3511-19aa87c53dd4
Pkg.status()

# ╔═╡ a588a9c0-d631-11ea-16ae-0f28a8a6be17
function get_data(batch_size)
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtrain = reshape(xtrain, 28^2, :)
    DataLoader(xtrain, ytrain, batchsize=batch_size, shuffle=true)
end

# ╔═╡ a69c3ca0-d631-11ea-0528-d1fc76e14cef
struct Encoder
    linear
    μ
    logσ
    Encoder(input_dim, latent_dim, hidden_dim, device) = new(
        Dense(input_dim, hidden_dim, tanh) |> device,   # linear
        Dense(hidden_dim, latent_dim) |> device,        # μ
        Dense(hidden_dim, latent_dim) |> device,        # logσ
    )
end

# ╔═╡ a9c39450-d631-11ea-33e2-bbc734ef92c0
function (encoder::Encoder)(x)
    h = encoder.linear(x)
    encoder.μ(h), encoder.logσ(h)
end

# ╔═╡ ae5689f0-d631-11ea-31ad-0dce3ca9e7f3
function reconstuct(encoder, decoder, x, device)
    μ, logσ = encoder(x)
    z = μ + device(randn(Float32, size(logσ))) .* exp.(logσ)
    μ, logσ, decoder(z)
end

# ╔═╡ e98125a2-d634-11ea-3efd-451d41f29f19
Decoder(input_dim, latent_dim, hidden_dim, device) = Chain(
    Dense(latent_dim, hidden_dim, tanh),
    Dense(hidden_dim, input_dim)
) |> device

# ╔═╡ b3045a8e-d631-11ea-27f2-895a6e478729
function model_loss(encoder, decoder, λ, x, device)
    μ, logσ, decoder_z = reconstuct(encoder, decoder, x, device)
    len = size(x)[end]
    # KL-divergence
    kl_q_p = 0.5f0 * sum(@. (exp(2f0 * logσ) + μ^2 -1f0 - 2f0 * logσ)) / len

    logp_x_z = -sum(logitbinarycrossentropy.(decoder_z, x)) / len
    # regularization
    reg = λ * sum(x->sum(x.^2), Flux.params(decoder))
    
    -logp_x_z + kl_q_p + reg
end

# ╔═╡ b6700e40-d631-11ea-3c08-37b106ce0f6d

function convert_to_image(x, y_size)
    Gray.(permutedims(vcat(reshape.(chunk(sigmoid.(x |> cpu), y_size), 28, :)...), (2, 1)))
end

# ╔═╡ b9a6cf40-d631-11ea-3ae6-15683043fe21
@with_kw mutable struct Args
    η = 1e-3                # learning rate
    λ = 0.01f0              # regularization paramater
    batch_size = 128        # batch size
    sample_size = 10        # sampling size for output    
    epochs = 20             # number of epochs
    seed = 0                # random seed
    cuda = true             # use GPU
    input_dim = 28^2        # image size
    latent_dim = 2          # latent dimension
    hidden_dim = 500        # hidden dimension
    verbose_freq = 10       # logging for every verbose_freq iterations
    tblogger = false        # log training with tensorboard
    save_path = "output"    # results path
end

# ╔═╡ be20bea0-d631-11ea-35d8-03ff21bef32a
function train(; kws...)
    # load hyperparamters
    args = Args(; kws...)
    args.seed > 0 && Random.seed!(args.seed)

    # GPU config
    if args.cuda && has_cuda_gpu()
        device = gpu
        @info "Training on GPU"
    else
        device = cpu
        @info "Training on CPU"
    end

    # load MNIST images
    loader = get_data(args.batch_size)
    
    # initialize encoder and decoder
    encoder = Encoder(args.input_dim, args.latent_dim, args.hidden_dim, device)
    decoder = Decoder(args.input_dim, args.latent_dim, args.hidden_dim, device)

    # ADAM optimizer
    opt = ADAM(args.η)
    
    # parameters
    ps = Flux.params(encoder.linear, encoder.μ, encoder.logσ, decoder)

    !ispath(args.save_path) && mkpath(args.save_path)

    # logging by TensorBoard.jl
    if args.tblogger
        tblogger = TBLogger(args.save_path, tb_overwrite)
    end

    # fixed input
    original, _ = first(get_data(args.sample_size^2))
    original = original |> device
    image = convert_to_image(original, args.sample_size)
    image_path = joinpath(args.save_path, "original.png")
    save(image_path, image)

    # training
    train_steps = 0
    @info "Start Training, total $(args.epochs) epochs"
    for epoch = 1:args.epochs
        @info "Epoch $(epoch)"
        progress = Progress(length(loader))

        for (x, _) in loader 
            loss, back = Flux.pullback(ps) do
                model_loss(encoder, decoder, args.λ, x |> device, device)
            end
            grad = back(1f0)
            Flux.Optimise.update!(opt, ps, grad)
            # progress meter
            next!(progress; showvalues=[(:loss, loss)]) 

            # logging with TensorBoard
            if args.tblogger && train_steps % args.verbose_freq == 0
                with_logger(tblogger) do
                    @info "train" loss=loss
                end
            end

            train_steps += 1
        end
        # save image
        _, _, rec_original = reconstuct(encoder, decoder, original, device)
        image = convert_to_image(rec_original, args.sample_size)
        image_path = joinpath(args.save_path, "epoch_$(epoch).png")
        #save(image_path, image)
        @info "Image saved: $(image_path)"
    end

    # save model
    model_path = joinpath(args.save_path, "model.bson") 
    let encoder = cpu(encoder), decoder = cpu(decoder), args=struct2dict(args)
        #BSON.@save model_path encoder decoder args
        @info "Model saved: $(model_path)"
    end
	return encoder, decoder, args
end

# ╔═╡ f37da220-d631-11ea-371c-b334d3cf5116
begin
	encoder1, decoder, args = train()
end

# ╔═╡ 2e3ed430-d662-11ea-26df-41444b5a2140
begin
	model_path=joinpath(args.save_path, "model.bson") 
	BSON.@save model_path encoder1 decoder args
end

# ╔═╡ 0c547842-d633-11ea-242f-cff2b1c7d4b6
function plot_result(encoder, decoder, args)
    device = args.cuda && has_cuda_gpu() ? gpu : cpu
    encoder, decoder = encoder |> device, decoder |> device
    # load MNIST images
    loader = get_data(args.batch_size)
	
    # clustering in the latent space
    # visualize first two dims
    #plt = scatter(palette=:rainbow)
	μ₁ = []
	μ₂ = []
	logσ = []
	y = []
    for (i, (x, yₙ)) in enumerate(loader)
        i < 20 || break
        μₙ, logσₙ = encoder(x |> device)
        append!(μ₁, μₙ[1,:])
		append!(μ₂, μₙ[2,:])
		append!(y,yₙ)
    end
	scatter(μ₁,μ₂,color=y, lab="",
		markerstrokewidth=0,
		markeralpha=0.8,
		aspect_ratio=1)
end

# ╔═╡ da4afaf0-d631-11ea-00b3-1945d1b3c8c1
cluster = plot_result(encoder1, decoder, args);

# ╔═╡ df619f7e-d631-11ea-1aa1-97d502069890
function plot_images(encoder, decoder, args, dims)
    device = args.cuda && has_cuda_gpu() ? gpu : cpu
    decoder = decoder |> device
    # load MNIST images


    #z = range(dims[1], stop=dims[2], length=4)
	z=[dims[1]]
    len = Base.length(z)
    z1 = repeat(z, len)
    z2 = sort(z1)
    x = zeros(Float32, args.latent_dim, len^2) |> device
    x[1, :] = dims[1]
    x[2, :] = dims[2]
    samples = decoder(x)
    image = convert_to_image(samples, len)
	plot(image)
end

# ╔═╡ 93b27ad0-d665-11ea-32bc-97e4e7edeee0
@bind dims html"""
<canvas width="240" height="200" style="position: relative"></canvas>

<script>
// 🐸 `this` is the cell output wrapper - we use it to select elements 🐸 //
const canvas = this.querySelector("canvas")
const ctx = canvas.getContext("2d")

var startX = 80
var startY = 40

function onmove(e){
	// 🐸 We send the value back to Julia 🐸 //
	canvas.value = [e.layerX/20 - 6 , -e.layerY/20 + 5]
	canvas.dispatchEvent(new CustomEvent("input"))

	ctx.fillStyle = '#ffecec'
	ctx.fillRect(0, 0, 240, 200)
	ctx.fillStyle = '#3f3d6d'
	//ctx.fillRect(startX,startY,e.layerX,e.layerY);
}

canvas.onmousedown = e => {
	startX = e.layerX
	startY = e.layerY
	canvas.onmousemove = onmove
}

canvas.onmouseup = e => {
	canvas.onmousemove = null
}

// Fire a fake mousemoveevent to show something
onmove({layerX: 130, layerY: 160})

</script>
"""

# ╔═╡ cfd699e0-d635-11ea-00b6-f10bd6d4aab6
plt = plot_images(encoder1, decoder, args, dims)

# ╔═╡ 5319bf00-d652-11ea-0682-73806a814465
md"""The dimensions are 

	x = $(dims[1])	y = $(dims[2])"""

# ╔═╡ c3635760-d640-11ea-29e0-01efffdb6357
begin 
	slider1 = @bind dim1 Slider(-6:0.2:6)
	field1 = @bind dim1 NumberField(-6:6)
	slider2 =@bind dim2 Slider(-6:0.2:6)
	field2 = @bind dim2 NumberField(-6:6)
md"""

X: $(slider1) $(field1)

Y: $(slider2) $(field2)
"""
end

# ╔═╡ 566062b0-d64b-11ea-269c-9f7e0f406973
function plot_on_cluster(plt, dims)
	plt2=deepcopy(plt)
	plot(plt2)
	scatter!([dims[1]],[dims[2]],markershape=:hexagon, color= "black",markersize=5,labels="selection")
end

# ╔═╡ 99f2dbb0-d64c-11ea-3789-d38a4a95e4e9
begin
	#dims = [dim1,dim2]
	plot_on_cluster(cluster, dims)

end

# ╔═╡ Cell order:
# ╠═1338fbd0-d62f-11ea-24b1-0fc6f3dec14b
# ╠═91e51d60-d62f-11ea-32c5-490645dfe7fe
# ╠═a42e4940-e2bb-11ea-3511-19aa87c53dd4
# ╠═819ebe60-d630-11ea-3c01-0b746169f32b
# ╠═a588a9c0-d631-11ea-16ae-0f28a8a6be17
# ╠═a69c3ca0-d631-11ea-0528-d1fc76e14cef
# ╠═a9c39450-d631-11ea-33e2-bbc734ef92c0
# ╠═ae5689f0-d631-11ea-31ad-0dce3ca9e7f3
# ╠═e98125a2-d634-11ea-3efd-451d41f29f19
# ╠═b3045a8e-d631-11ea-27f2-895a6e478729
# ╠═b6700e40-d631-11ea-3c08-37b106ce0f6d
# ╠═b9a6cf40-d631-11ea-3ae6-15683043fe21
# ╟─be20bea0-d631-11ea-35d8-03ff21bef32a
# ╠═f37da220-d631-11ea-371c-b334d3cf5116
# ╠═2e3ed430-d662-11ea-26df-41444b5a2140
# ╠═0c547842-d633-11ea-242f-cff2b1c7d4b6
# ╟─da4afaf0-d631-11ea-00b3-1945d1b3c8c1
# ╟─df619f7e-d631-11ea-1aa1-97d502069890
# ╟─cfd699e0-d635-11ea-00b6-f10bd6d4aab6
# ╟─93b27ad0-d665-11ea-32bc-97e4e7edeee0
# ╟─5319bf00-d652-11ea-0682-73806a814465
# ╟─99f2dbb0-d64c-11ea-3789-d38a4a95e4e9
# ╠═c3635760-d640-11ea-29e0-01efffdb6357
# ╠═566062b0-d64b-11ea-269c-9f7e0f406973
