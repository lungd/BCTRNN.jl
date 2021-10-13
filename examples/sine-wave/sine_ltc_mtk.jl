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
  T=Float32, in_size=10, model_size=5,
  tgrad=false, gen_jac=false, lr=0.02, kwargs...)

  train_dl = generate_2d_data(T)

  f_in = in_size
  f_out = 1

  input_matrices = Dict(
      :w_sens => ones(T, f_in, model_size),
      :E_sens => ones(T, f_in, model_size)
  )
  syns_matrices = Dict(
      :w_syns => ones(T, model_size, model_size),
      :E_syns => ones(T, model_size, model_size)
  )
  im = BCTRNN.InputMatrices(input_matrices)
  sm = BCTRNN.SynsMatrices(syns_matrices)
  
  im = BCTRNN.InputAllToAll()
  sm = BCTRNN.SynsFullyConnected()
  om = BCTRNN.OutputAll()
  wiring = BCTRNN.WiringConfig(f_in, model_size, im,sm,om)
  @show wiring.s_in
  @show wiring.n_total

  cell,defs,sys = BCTRNN.LTCMTK(wiring, solver, sensealg; T, tgrad, gen_jac, kwargs...)
  model = FastChain(#(x,p)->ones(eltype(x), f_in, size(x)[2]),
                    FastDense(2, in_size),
                    cell,
                    FastDense(wiring.n_out, f_out))

  hs = []
  for (k,v) in wiring.matrices
    push!(hs, heatmap(v, title=k))
  end
  display(plot(hs..., layout=length(hs)))

  cb = BCTRNN.MyCallback(T; ecb=mycb, nepochs=epochs, nsamples=length(train_dl))
  #opt = GalacticOptim.Flux.Optimiser(ClipValue(0.5), ADAM(0.02))
  opt = BCTRNN.ClampBoundOptim(BCTRNN.get_bounds(model,T)..., ClipValue(T(1.0)), ADAM(T(lr)))
  BCTRNN.optimize(model, BCTRNN.loss_seq, cb, opt, train_dl, epochs, T), model, defs, sys
end

@time res, model, defs, sys = train_sine_fc(200, TRBDF2(); lr=0.02, in_size=5, model_size=8,
save_everystep=true, save_start=true, save_end=true, tgrad=true, gen_jac=false,
abstol=1e-5, reltol=1e-5,
)







function train_sine(epochs, solver=Tsit5();
  sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true)),
  T=Float32, 
  n_sensory=2, n_inter=4, n_command=4, n_motor=1,
  sensory_inter=2, inter_command=3, command_command=2, command_motor=2,
  tgrad=true, gen_jac=false, cv=1.0, lr=0.02, kwargs...)

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
                    BCTRNN.DLeak(wiring, solver, sensealg; T, tgrad, gen_jac, kwargs...),
                    FastDense(wiring.n_out, f_out))

  hs = []
  for (k,v) in wiring.matrices
    push!(hs, heatmap(v, title=k))
  end
  display(plot(hs..., layout=length(hs)))

  cb = BCTRNN.MyCallback(T; ecb=mycb, nepochs=epochs, nsamples=length(train_dl))
  #opt = GalacticOptim.Flux.Optimiser(ClipValue(0.5), ADAM(0.02))
  opt = BCTRNN.ClampBoundOptim(BCTRNN.get_bounds(model,T)..., ClipValue(T(cv)), ADAM(T(lr)))
  BCTRNN.optimize(model, BCTRNN.loss_seq, cb, opt, train_dl, epochs, T), model
end


#  60.991400 seconds (168.50 M allocations: 19.765 GiB, 5.07% gc time)
@time train_sine(200, Tsit5(), lr=0.03,
n_sensory=5, n_inter=10, n_command=10, n_motor=5,
sensory_inter=2, inter_command=5, command_command=3, command_motor=5,)