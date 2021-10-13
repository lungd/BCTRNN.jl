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
                    BCTRNN.LTCSynStateMTK(wiring, solver, sensealg; T, mtkize, gen_jac, kwargs...),
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

@time train_sine_ss_fc(100, Vern7(); model_size=8, mtkize=false, lr=0.01,
)





function train_sine_ss_ncp(epochs, solver=Tsit5();
  sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
  T=Float32, 
  n_sensory=2, n_inter=5, n_command=5, n_motor=2,
  sensory_inter=2, inter_command=3, command_command=2, command_motor=2,
  mtkize=false, gen_jac=false, lr=0.02, kwargs...)

  train_dl = generate_2d_data(T)

  f_in = 2
  f_out = 1

  im = BCTRNN.InputAllToFirstN(n_sensory)
  sm = BCTRNN.SynsNCP(; n_sensory, n_inter, n_command, n_motor,
    sensory_inter, inter_command, command_command, command_motor
  )
  om = BCTRNN.OutputAll()
  om = BCTRNN.OutputIdxs(collect(1:n_sensory+n_inter+n_command+n_motor))
  wiring = BCTRNN.WiringConfig(f_in, im,sm,om)

  model = FastChain(BCTRNN.Mapper(f_in),
                    BCTRNN.LTCSynStateMTK(wiring, solver, sensealg; T, mtkize, gen_jac, kwargs...),
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

@time train_sine_ss_ncp(100, ROCK4(), n_sensory=5, n_inter=10, n_command=10, mtkize=false, lr=0.01,
)


@time train_sine_ss_ncp(100, TRBDF2(linsolve=LinSolveGMRES()), n_sensory=3, n_inter=5, n_command=5, mtkize=false, lr=0.01,
)