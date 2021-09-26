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
  m.input = x
  m.state, y = m.cell(m.state, m.input, x, p)
  return y
end
Base.show(io::IO, m::MyRecur) = print(io, "MyRecur(", m.cell, ")")
initial_params(m::MyRecur) = initial_params(m.cell)
paramlength(m::MyRecur) = length(m.p)
Flux.functor(m::MyRecur) = (m.p), re -> MyRecur(m.cell, m.state, m.input, re...)
Flux.trainable(m::MyRecur) = (m.p,)
get_bounds(m::MyRecur, T::DataType=eltype(m.state)) = get_bounds(m.cell, T)
Flux.reset!(m::MyRecur, p=m.p) = (m.state = reshape(p[end-length(m.cell.state0)+1:end],:,1))
function reset_state!(m::MyRecur, p=m.p)
  #m.p = p
  pl = length(p)
  state = @view p[reshape(pl-size(m.cell.state0,1)+1:pl, :, 1)]
  m.state = state
  m.input = zero(state)
  nothing
end
# TODO: reset_state! for cell with train_u0=false


struct BCTRNNCell{SOLVER,SENSE,PROB,LB,UB,S,P}
  n_in::Int
  n_sens::Int
  n_neurons::Int
  n_out::Int
  solver::SOLVER
  sensealg::SENSE
  prob::PROB
  lb::LB
  ub::UB
  state0::S
  p::P

  function BCTRNNCell(n_in, n_sens, n_neurons, n_out, solver, sensealg, prob, lb, ub, state0, p)
    new{typeof(solver),typeof(sensealg),typeof(prob),typeof(lb),typeof(ub),typeof(state0),typeof(p)}(
                       n_in, n_sens, n_neurons, n_out, solver, sensealg, prob, lb, ub, state0, p)
  end
end




function (m::BCTRNNCell)(h, last_input, x::AbstractArray, p=m.p)
  # size(h) == (N,1) at the first MTKNODECell invocation. Need to duplicate batchsize times
  T = eltype(p)
  nstates = size(h,1)
  batchsize = size(x,2)
  num_reps = batchsize-size(h,2)+1
  u0s = repeat(h, 1, num_reps)

  interpol = DataInterpolations.LinearInterpolation([last_input, x], 0f0:1f0)

  p_ode = @view p[1:end-nstates]
  #p_ode_ca = ltc_sys_vec2ca(m.n_in, m.n_neurons, (@view p[1:end-nstates]))
  
  prob = m.prob

  #dosetimes = collect(0.1:0.2:0.9)
  #condition(u,t,integrator) = t ∈ dosetimes
  condition2(u,t,integrator) = true
  

  function prob_func(prob,i,repeat)
    u0 = @view u0s[:,i]
    #p = ComponentArray(stim=(@view x[:,i]), p_ode_ca)
    p = vcat((@view x[:,i]), p_ode)

    affect!(integrator) = integrator.p[1:m.n_in] .= @view interpol(integrator.t)[:,i]
    #cb = DiscreteCallback(condition,affect!)
    cb = ContinuousCallback(condition2, affect!)

    remake(prob; u0, p, callback=cb)
  end

  function output_func(sol,i)
    sol.retcode != :Success && return fill(T(Inf), nstates), false
    sol[:, end], false
  end
  
  

  ensemble_prob = EnsembleProblem(prob; prob_func, output_func, safetycopy=false) # TODO: safetycopy ???
  sol = solve(ensemble_prob, m.solver, EnsembleThreads(), trajectories=batchsize;
              save_end=true, save_everystep=false, save_start=false,
              sensealg=m.sensealg,
              reltol=1e-3, abstol=1e-4,
              #dt=0.2, adaptive=false, dense=false,
  )
  sa = Array(sol)
  return (sa, (@view sa[end-m.n_out+1:end, :]))
end



Base.show(io::IO, m::BCTRNNCell) = print(io, "BCTRNNCell(", m.n_in, ",", m.n_out, ")")
initial_params(m::BCTRNNCell) = m.p
paramlength(m::BCTRNNCell) = m.paramlength
Flux.functor(m::BCTRNNCell) = (m.p,), re -> BCTRNNCell(m.n_in, m.n_sens, m.n_neurons, m.n_out, m.solver, m.sensealg, m.prob, m.lb, m.ub, m.state0, re...)
Flux.trainable(m::BCTRNNCell) = (m.p,)


function get_bounds(m::BCTRNNCell, T::DataType=eltype(m.p))
  m.lb, m.ub
end