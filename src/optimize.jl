mutable struct ClampBoundOptim{V} <: Flux.Optimise.AbstractOptimiser
  os::Vector{Any}
  lb::V
  ub::V
end
ClampBoundOptim(lb,ub,o...) = ClampBoundOptim{typeof(lb)}(Any[o...],lb,ub)

Flux.@forward ClampBoundOptim.os Base.getindex, Base.first, Base.last, Base.lastindex, Base.push!, Base.setindex!
Flux.@forward ClampBoundOptim.os Base.iterate

Base.getindex(c::ClampBoundOptim, i::AbstractArray) = ClampBoundOptim(c.lb,c.ub,c.os[i]...)

function Flux.Optimise.apply!(o::ClampBoundOptim{V}, x, Δ) where V
  for opt in o.os
    Δ = Flux.Optimise.apply!(opt, x, Δ)
  end
  return Δ
end

function Flux.Optimise.update!(opt::ClampBoundOptim{V}, x, x̄) where V
  x̄r = Flux.Optimise.ArrayInterface.restructure(x, x̄) # address some cases where Zygote's
                                          # output are not mutable, see #1510
  x .-= Flux.Optimise.apply!(opt, x, x̄r)
  x .= map(i -> clamp(x[i], opt.lb[i], opt.ub[i]), 1:length(x))
end


load_model(chain::Flux.Chain, T::DataType=Float32) = (Flux.destructure(chain)..., get_bounds(chain,T)...)
load_model(chain::FastChain, T::DataType=Float32) = (initial_params(chain), chain, get_bounds(chain,T)...)
load_model(model, T::DataType=Float32) = (initial_params(model), model, get_bounds(model,T)...)

function optimize(chain, loss, cb, opt, train_dl, epochs=1, T::DataType=Float32, AD=GalacticOptim.AutoZygote())
  pp, model, lb, ub = load_model(chain, T)

  println("--------------- optimize ---------------")
  println("# training samples:         $(length(train_dl))")
  println("# parameters:               $(length(pp))")
  println("typeof(p):                  $(typeof(pp))")
  println("# epochs:                   $(epochs)")
  println("# lb:                       $(length(lb))")
  println("# ub:                       $(length(ub))")
  println("typeof(lb):                 $(typeof(lb))")


  # mycb = LTC.MyCallback(T, cb, epochs, length(train_dl))
  train_dlnc = epochs > 1 ? ncycle(train_dl, epochs) : train_dl


  f = (θ,p,x,y) -> loss(θ,model,x,y)
  optfun = GalacticOptim.OptimizationFunction(f, AD)
  optfunc = GalacticOptim.instantiate_function(optfun, pp, AD, nothing)
  optprob = GalacticOptim.OptimizationProblem(optfunc, pp, lb=lb, ub=ub,
                                grad = true, hess = true, #sparse = true,
                                #parallel=ModelingToolkit.MultithreadedForm()
                                )

  sol = GalacticOptim.solve(optprob, opt, train_dlnc, cb = cb)

  # optfun = GalacticOptim.OptimizationFunction(f, AD)
  # optfunc = GalacticOptim.instantiate_function(optfun, sol.u, AD, nothing)
  # optprob = GalacticOptim.OptimizationProblem(optfunc, sol.u, lb=lb, ub=ub,
  #                               grad = true, hess = true, #sparse = true,
  #                               #parallel=ModelingToolkit.MultithreadedForm()
  #                               )

  # sol = GalacticOptim.solve(optprob, BFGS(initial_stepnorm=0.01), train_dlnc, cb = cb)

  for i in 1:length(pp)
    pp[i] == sol[i] && println(i)
  end
  println(pp)
  println(sol)
  sol
end
