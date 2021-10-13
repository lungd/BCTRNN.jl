mutable struct MyRecur{T,S,I,P}
  cell::T
  state::S
  input::I
  p::P
end
function MyRecur(cell)
  p = initial_params(cell)
  MyRecur(cell, cell.state0, zero(cell.state0), p)
end
function (m::MyRecur)(x, p=m.p)
  batchsize = size(x,2)
  m.input = x
  h′, y = m.cell(m.state, m.input, x, p)
  Inf ∈ h′ && return fill(Inf32, size(y,1), batchsize)
  m.state = h′
  return y
end
# function (m::MyRecur)(x::AbstractArray{T, 3}, p=m.p) where T
#   h = [m(view(x, :, :, i), p) for i in 1:size(x, 3)]
#   sze = size(h[1])
#   reshape(reduce(hcat, h), sze[1], sze[2], length(h))
# end
Base.show(io::IO, m::MyRecur) = print(io, "MyRecur(", m.cell, ")")
initial_params(m::MyRecur) = initial_params(m.cell)
paramlength(m::MyRecur) = length(m.p)
Flux.functor(::Type{<:MyRecur}, m) = (m.p), re -> MyRecur(m.cell, m.state, m.input, re...)
Flux.trainable(m::MyRecur) = (m.p,)
get_bounds(m::MyRecur, T::DataType=eltype(m.state)) = get_bounds(m.cell, T)
Flux.reset!(m::MyRecur) = reset_state!(m, m.p)
@views function reset_state!(m::MyRecur, p=m.p)
  #m.p = p
  pl = length(p)
  state = p[reshape(pl-size(m.cell.state0,1)+1:pl, :, 1)]
  m.state = state
  m.input = reshape(zeros(eltype(m.input), size(state,1)), :, 1)
  nothing
end
# TODO: reset_state! for cell with train_u0=false

function Base.getproperty(m::MyRecur{T,S,I,P}, s::Symbol) where {T,S,I,P}
  if s === :cell
    return getfield(m, :cell)
  elseif s === :state
    return getfield(m, :state)
  elseif s === :input
    return getfield(m, :input)
  elseif s === :p
    return getfield(m, :p)
  else
    return getfield(m, s)
  end
end

ode_solve_kwargs(; abstol=1e-4, reltol=1e-3, save_everystep=false, save_start=false, save_end=true, kwargs...) = 
  (abstol=abstol, reltol=reltol, save_everystep=save_everystep, save_start=save_start, save_end=save_end, kwargs...)

struct BCTRNNCell{W,SOLVER,SENSE,PROB,LB,UB,S,P,KW}
  wiring::W

  solver::SOLVER
  sensealg::SENSE
  prob::PROB
  lb::LB
  ub::UB
  state0::S
  p::P
  kwargs::KW

  function BCTRNNCell(wiring, solver, sensealg, prob, lb, ub, state0, p; kwargs...)
    new{typeof(wiring),typeof(solver),typeof(sensealg),typeof(prob),typeof(lb),typeof(ub),typeof(state0),typeof(p), typeof(kwargs)}(
                      wiring, solver, sensealg, prob, lb, ub, state0, p, kwargs)
  end
end

function Base.getproperty(m::BCTRNNCell{W,SOLVER,SENSE,PROB,LB,UB,S,P,KW}, s::Symbol) where {W,SOLVER,SENSE,PROB,LB,UB,S,P,KW}
  if s === :s_in
    return getproperty(getfield(m, :wiring), s)
  elseif s === :n_total
    return getproperty(getfield(m, :wiring), s)
  # elseif s === :n_in
  #   return getproperty(getfield(m, :wiring), s)
  elseif s === :n_out
    return getproperty(getfield(m, :wiring), s)
  elseif s === :wiring
    return getfield(m, :wiring)
  elseif s === :solver
    return getfield(m, :solver)
  elseif s === :sensealg
    return getfield(m, :sensealg)
  elseif s === :prob
    return getfield(m, :prob)
  elseif s === :lb
    return getfield(m, :lb)
  elseif s === :ub
    return getfield(m, :ub)
  elseif s === :state0
    return getfield(m, :state0)
  elseif s === :p
    return getfield(m, :p)
  elseif s === :kwargs
    return getfield(m, :kwargs)
  else
    return getfield(m, s)
  end
end

function BCTRNNCell(wiring, solver, sensealg, prob, lb, ub; kwargs...)
  u0 = prob.u0
  p = prob.p
  p_ode = p[wiring.s_in+1:end]
  state0 = reshape(u0, :, 1)
  θ = vcat(p_ode, u0)
  BCTRNNCell(wiring, solver, sensealg, prob, lb, ub, state0, θ; kwargs...)
end

function BCTRNNCell(wiring, solver, sensealg, odef, u0, tspan, p, lb, ub; mtkize=false, gen_jac=false, kwargs...)
  _prob = ODEProblem{true}(odef, u0, tspan, p)
  p_ode = @view p[wiring.s_in+1:end]
  prob = mtkize_prob(_prob, odef, u0, tspan, p, mtkize, gen_jac)

  # eqs = ModelingToolkit.equations(sys)
  # sts = ModelingToolkit.states(sys)
  # noiseeqs = 0.1f0 .* sts
  # ps = ModelingToolkit.parameters(sys)
  # @named sde = SDESystem(eqs,noiseeqs,ModelingToolkit.get_iv(sys),sts,ps)
  # prob = SDEProblem(sde,u0,tspan,p)

  state0 = reshape(u0, :, 1)
  θ = vcat(p_ode, u0)

  BCTRNNCell(wiring, solver, sensealg, prob, lb, ub, state0, θ; kwargs...)
end



function (m::BCTRNNCell)(h, _last_input, x::AbstractArray, p=m.p)
  # size(h) == (N,1) at the first MTKNODECell invocation. Need to duplicate batchsize times
  nstates = size(h,1)
  batchsize = size(x,2)
  num_reps = batchsize-size(h,2)+1
  u0s = repeat(h, 1, num_reps)
  last_input = repeat(_last_input, 1, num_reps)

  tspan_h = m.prob.tspan
  tspan_o = (tspan_h[2], tspan_h[2]+1)

  p_ode = @view p[1:end-nstates]
  

  out_idxs = _output_idxs(m.wiring.output_mapping, m.wiring.n_total)
  

  #sol = solve_ode(m, u0s, tspan_h, p_ode, last_input)
  sol = solve_ode(m, u0s, tspan_h, p_ode, x)
  h′ = sol[:, end, :]
  
  
  Inf ∈ h′ && return (fill(Inf32, nstates, batchsize), fill(Inf32, length(out_idxs), batchsize))
  NaN ∈ h′ && return (fill(Inf32, nstates, batchsize), fill(Inf32, length(out_idxs), batchsize))
  
  return (h′, (@view h′[out_idxs, :]))

  # sol = solve_ode(m, h′, tspan_o, p_ode, x)
  # h′ = sol[:, end, :]

  # return (h′, h′[out_idxs, :])
end


function solve_ode(m, u0s, tspan, p_ode, x)
  batchsize = size(u0s, 2)
  prob = m.prob
  function prob_func(prob,i,repeat)
    u0 = @view u0s[:,i]
    stim = @view x[:,i]
    p = vcat(stim, p_ode)

    # condition(u,t,integrator) = any((u .> 10) .| (u .< -10))
    # function affect!(integrator)
    #   for ii in 1:length(integrator.u)
    #     integrator.u[ii] = clamp(integrator.u[ii], -one(integrator.u[ii]), one(integrator.u[ii]))
    #   end
    # end
    # cb = ContinuousCallback(condition,affect!)

    remake(prob; u0, tspan, p)
  end
  function output_func(sol,i)

    # Zygote.@ignore begin 
    #   if i == 1 && length(sol.t) > 1
    #     @show length(sol.t)
    #     fig = plot(xlabel="t", ylabel="step size",title="Adaptive step size")
    #     steps1 = (@view sol.t[2:end]) - (@view sol.t[1:end-1])
    #     plot!(fig, (@view sol.t[2:end]),steps1)
    #     display(fig)
    #   end
    # end
    (sol.retcode != :Success || NaN ∈ sol) && return fill(Inf32, size(u0s,1)), false
    (@view sol[:, end]), false
  end

  kwargs = ode_solve_kwargs(; m.kwargs...)
  ensemble_prob = EnsembleProblem(prob; prob_func, output_func, safetycopy=false) # TODO: safetycopy ???
  sa = Array(solve(ensemble_prob, m.solver, EnsembleThreads(), trajectories=batchsize;
              sensealg=m.sensealg,
              kwargs...
  ))

  sa
end


Base.show(io::IO, m::BCTRNNCell) = print(io, "BCTRNNCell(", m.s_in, ",", m.n_out, ")")
initial_params(m::BCTRNNCell) = m.p
paramlength(m::BCTRNNCell) = length(m.p)
Flux.functor(::Type{<:BCTRNNCell}, m) = (m.p,), re -> BCTRNNCell(m.wiring, m.solver, m.sensealg, m.prob, m.lb, m.ub, m.state0, re...)
Flux.trainable(m::BCTRNNCell) = (m.p,)


function get_bounds(m::BCTRNNCell, T::DataType=eltype(m.p))
  m.lb, m.ub
end

function mtkize_prob(prob, f, u0, tspan, p, mtkize, gen_jac)
  mtkize == false && return prob
  
  sys = ModelingToolkit.modelingtoolkitize(prob)
  #sys = ModelingToolkit.structural_simplify(sys)

  gen_jac == false && return ODEProblem{true}(sys,u0,tspan,p, tgrad=true)
  
  jac = eval(ModelingToolkit.generate_jacobian(sys)[2])
  odef = ODEFunction{true}(f, jac=jac)
  ODEProblem{true}(odef,u0,tspan,p, tgrad=true)
end
