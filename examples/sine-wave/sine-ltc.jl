using BCTRNN
using DiffEqSensitivity
using OrdinaryDiffEq
import DiffEqFlux: FastChain, FastDense
import Flux: ClipValue, ADAM

# Not in Project.toml
using Plots
gr()

include("sine_wave_dataloader.jl")

function train_sine(epochs, solver=Tsit5();
  sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
  T=Float32, model_size=5,
  kwargs...)

  train_dl = generate_2d_data(T)

  f_in = 2
  f_out = 1
  n_neurons = model_size
  n_sens = n_neurons
  n_out = n_neurons

  model = FastChain(BCTRNN.Mapper(f_in),
                    BCTRNN.LTC(f_in, n_neurons, solver, sensealg; n_sens, n_out),
                    FastDense(n_out, f_out))

  cb = BCTRNN.MyCallback(T; cb=mycb, ecb=(_)->nothing, nepochs=epochs, nsamples=length(train_dl))
  #opt = GalacticOptim.Flux.Optimiser(ClipValue(0.5), ADAM(0.02))
  opt = BCTRNN.ClampBoundOptim(BCTRNN.get_bounds(model,T)..., ClipValue(T(1.0)), ADAM(T(0.02)))
  BCTRNN.optimize(model, BCTRNN.loss_seq, cb, opt, train_dl, epochs, T), model
end


#  60.991400 seconds (168.50 M allocations: 19.765 GiB, 5.07% gc time)
@time train_sine(200, Tsit5(); model_size=3)
@time train_sine(100, Tsit5(); model_size=10)

@time train_sine(100, AutoTsit5(Rosenbrock23(autodiff=false)); model_size=10)
@time train_sine(20, AutoTsit5(Rosenbrock23(autodiff=false)); model_size=30)

#@time train_sine(20, SOSRA(); model_size=20)
#@time train_sine(200, SRA3(); model_size=7)