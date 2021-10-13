using BCTRNN
using DiffEqSensitivity
using OrdinaryDiffEq
import DiffEqFlux: FastChain, FastDense
import Flux: ClipValue, ADAM

# Not in Project.toml
using Plots
gr()

include("sine_wave_dataloader.jl")


function train_sine_ss_fc(epochs, solver=Tsit5();
  sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
  T=Float32, model_size=5,
  mtkize=false, gen_jac=false, lr=0.02, kwargs...)

  train_dl = generate_2d_data(T)

  f_in = 2
  f_out = 1

  im = BCTRNN.InputAllToAll()
  sm = BCTRNN.SynsFullyConnected()
  #om = BCTRNN.OutputAll()
  om = BCTRNN.OutputIdxs(collect(1:model_size))
  wiring = BCTRNN.WiringConfig(f_in, model_size, im,sm,om)

  model = FastChain(BCTRNN.Mapper(f_in),
                    BCTRNN.LTCSynState(wiring, solver, sensealg; T, mtkize, gen_jac, kwargs...),
                    FastDense(wiring.n_out, f_out))

  hs = []
  for (k,v) in wiring.matrices
    push!(hs, heatmap(v, title=k))
  end
  display(plot(hs..., layout=length(hs)))

  cb = BCTRNN.MyCallback(T; ecb=mycb, nepochs=epochs, nsamples=length(train_dl))
  #opt = GalacticOptim.Flux.Optimiser(ClipValue(0.5), ADAM(0.02))
  opt = BCTRNN.ClampBoundOptim(BCTRNN.get_bounds(model,T)..., ClipValue(T(1.0)), ADAM(T(lr)))
  BCTRNN.optimize(model, BCTRNN.loss_seq, cb, opt, train_dl, epochs, T), model
end

@time train_sine_ss_fc(1000, AutoTsit5(Rosenbrock23(autodiff=false)); model_size=5, mtkize=true, lr=0.004)

