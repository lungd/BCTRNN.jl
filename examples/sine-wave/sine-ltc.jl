using BCTRNN
using DiffEqSensitivity
using OrdinaryDiffEq
import DiffEqFlux: FastChain, FastDense
import Flux: ClipValue, ADAM

# Not in Project.toml
using Plots
gr()

include("sine_wave_dataloader.jl")


function train_sine_fc(epochs, solver=Tsit5();
  sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
  T=Float32, model_size=5,
  mtkize=false, lr=0.02, kwargs...)

  train_dl = generate_2d_data(T)

  f_in = 2
  f_out = 1

  im = BCTRNN.InputAllToAll()
  sm = BCTRNN.SynsFullyConnected()
  om = BCTRNN.OutputAll()
  wiring = BCTRNN.WiringConfig(f_in, model_size, im,sm,om)

  model = FastChain(BCTRNN.Mapper(f_in),
                    BCTRNN.LTC(wiring, solver, sensealg; T, mtkize, kwargs...),
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

@time train_sine_fc(200, Tsit5(); model_size=10, mtkize=true)


function train_sine(epochs, solver=Tsit5();
  sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
  T=Float32, 
  n_sensory=2, n_inter=5, n_command=5, n_motor=1,
  sensory_inter=2, inter_command=3, command_command=2, command_motor=2,
  mtkize=false, lr=0.02, kwargs...)

  train_dl = generate_2d_data(T)

  f_in = 2
  f_out = 1

  im = BCTRNN.InputAllToFirstN(n_sensory)
  sm = BCTRNN.SynsNCP(; n_sensory, n_inter, n_command, n_motor,
    sensory_inter, inter_command, command_command, command_motor
  )
  om = BCTRNN.OutputAll()
  wiring = BCTRNN.WiringConfig(f_in, im,sm,om)

  model = FastChain(BCTRNN.Mapper(f_in),
                    BCTRNN.LTC(wiring, solver, sensealg; T, mtkize, kwargs...),
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


#  60.991400 seconds (168.50 M allocations: 19.765 GiB, 5.07% gc time)
@time train_sine(100, Tsit5(), mtkize=false) #  36.071273 seconds (190.71 M allocations: 9.256 GiB, 5.49% gc time)
@time train_sine(100, Tsit5(), n_sensory=3, n_motor=4, mtkize=false)
@time train_sine(200, Tsit5(); model_size=3, abstol=1e-4, reltol=1e-3)

@time train_sine(100, AutoTsit5(Rosenbrock23(autodiff=false)); model_size=10)
@time train_sine(20, AutoTsit5(Rosenbrock23(autodiff=false)); model_size=30)

#@time train_sine(20, SOSRA(); model_size=20)
#@time train_sine(200, SRA3(); model_size=7)