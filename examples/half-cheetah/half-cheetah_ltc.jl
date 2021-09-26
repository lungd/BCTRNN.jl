using BCTRNN
using DiffEqSensitivity
using OrdinaryDiffEq
import DiffEqFlux: FastChain, FastDense
import Flux: ClipValue, ADAM

# Not in Project.toml
using Plots
gr()

include("half_cheetah_data_loader.jl")

function train_cheetah(epochs, solver=nothing; sensealg=nothing,
  T=Float32, model_size=5, batchsize=1, seq_len=32, normalise=true,
  kwargs...)

  train_dl, test_dl, _, _ = get_2d_dl(T; batchsize, seq_len, normalise=true)
  @show size(first(train_dl)[1])
  @show size(first(train_dl)[1][1])
  
  f_in = 17
  f_out = 17
  n_neurons = model_size
  n_sens = n_neurons
  n_out = n_neurons

  model = FastChain(BCTRNN.Mapper(f_in),
                    BCTRNN.LTC(f_in, n_neurons, solver, sensealg; n_sens, n_out),
                    FastDense(n_out, f_out))

  cb = BCTRNN.MyCallback(T; cb=mycb, ecb=(_)->nothing, nepochs=epochs, nsamples=length(train_dl))
  #opt = GalacticOptim.Flux.Optimiser(ClipValue(0.5), ADAM(0.02))
  opt = BCTRNN.ClampBoundOptim(BCTRNN.get_bounds(model,T)..., ClipValue(T(1.0)), ADAM(T(0.01)))
  BCTRNN.optimize(model, BCTRNN.loss_seq, cb, opt, train_dl, epochs, T), model
end
#1173.351351 seconds (1.02 G allocations: 65.414 GiB, 1.82% gc time, 0.51% compilation time)
train_cheetah(30, AutoTsit5(Rosenbrock23(autodiff=false)); sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)), model_size=10, batchsize=12, abstol=1e-4, reltol=1e-4
)