abstract type InputMapping end
struct InputAllToAll <: InputMapping
end
struct InputAllToFirstN <: InputMapping
  n::Int
end
struct InputDiag <: InputMapping
end

function generate_sens_matrices(im::InputMapping, s_in, n_total; T=Float32)
  _w = zeros(T, s_in, n_total)
  _E = zeros(T, s_in, n_total)
  for (src,dst) in _input_idxs(im, s_in, n_total)
    add_synapse!(src, dst, _w, _E)
  end
  _w, _E
end

_input_idxs(m::InputAllToAll, s_in, n_total)    = [(src, dst) for dst in 1:n_total, src in 1:s_in]
_input_idxs(m::InputAllToFirstN, s_in, n_total) = [(src, dst) for dst in 1:m.n, src in 1:s_in]
_input_idxs(m::InputDiag, s_in, n_total)        = [(src, src) for src in 1:s_in]

get_n_in(m::InputMapping, n_total) = n_total
get_n_in(m::InputAllToFirstN, n_total) = m.n

abstract type OutputMapping end

struct OutputAll <: OutputMapping
end
struct OutputLastN <: OutputMapping
  n::Int
end
struct OutputIdxs <: OutputMapping
  idxs::Vector{Int}
end
_output_idxs(m::OutputAll, n_total)   = 1:n_total
_output_idxs(m::OutputLastN, n_total) = n_total-m.n+1:n_total
_output_idxs(m::OutputIdxs, n_total)  = m.idxs

get_n_out(m::OutputAll, n_total) = n_total
get_n_out(m::OutputLastN, n_total) = m.n
get_n_out(m::OutputIdxs, n_total) = length(m.idxs)

abstract type SynsMapping end

struct SynsFullyConnected <: SynsMapping
end

struct SynsNCP <: SynsMapping
  n_sensory::Int
  n_inter::Int
  n_command::Int
  n_motor::Int

  sensory_inter::Int
  inter_command::Int
  command_command::Int
  command_motor::Int
end

SynsNCP(; n_sensory=2, n_inter=5, n_command=5, n_motor=1,
  sensory_inter=2, inter_command=3, command_command=2, command_motor=2
) = SynsNCP(n_sensory, n_inter, n_command, n_motor,
    sensory_inter, inter_command, command_command, command_motor)

function generate_syns_matrices(sm::SynsMapping, n_total::Integer; T=Float32)
  _w = zeros(T, n_total, n_total)
  _E = zeros(T, n_total, n_total)
  for (src,dst) in _syns_idxs(sm, n_total)
    add_synapse!(src, dst, _w, _E)
  end
  _w, _E
end

_syns_idxs(m::SynsFullyConnected, n_total) = [(src, dst) for dst in 1:n_total, src in 1:n_total]

function _syns_idxs(m::SynsNCP, n_total) 
  idxs = Vector{Tuple{Int,Int}}()

  n_sensory       = m.n_sensory
  n_inter         = m.n_inter
  n_command       = m.n_command
  n_motor         = m.n_motor
  sensory_inter   = m.sensory_inter
  inter_command   = m.inter_command
  command_command = m.command_command
  command_motor   = m.command_motor

  #n_neurons = n_sensory + n_inter + n_command + n_motor
  n_neurons = n_total

  s = 1
  r_sens = s:s+n_sensory-1
  s += n_sensory
  r_inter = s:s+n_inter-1
  s += n_inter
  r_command = s:s+n_command-1
  s += n_command
  r_motor = s:s+n_motor-1

  w_syns = zeros(Float32, n_neurons, n_neurons)
  E_syns = zeros(Float32, n_neurons, n_neurons)

  connect_layers!(r_sens,    r_inter,   w_syns, E_syns; conns=sensory_inter)
  connect_layers!(r_inter,   r_command, w_syns, E_syns; conns=inter_command)
  connect_layers!(r_command, r_command, w_syns, E_syns; conns=command_command, handle_unconnected=false)
  connect_layers!(r_command, r_motor,   w_syns, E_syns; conns=command_motor)

  for dst in 1:size(w_syns,2)
    for src in 1:size(w_syns,1)
      w_syns[src,dst] == 0 && continue
      push!(idxs, (src,dst))
    end
  end
  
  idxs
end



mutable struct WiringConfig{T<:AbstractFloat, IM<:InputMapping, SM<:SynsMapping, OM<:OutputMapping}
  s_in::Int
  n_total::Int
  matrices::Dict{Symbol, Matrix{T}}
  input_mapping::IM
  syns_mapping::SM
  output_mapping::OM
end

WiringConfig(s_in::Integer, im::InputMapping, sm::SynsNCP, om::OutputMapping) = 
  WiringConfig(s_in, sm.n_sensory+sm.n_inter+sm.n_command+sm.n_motor, im, sm, om)

  function WiringConfig(s_in::Integer, n_total::Integer, im::InputMapping, sm::SynsMapping, om::OutputMapping)
  w_sens, E_sens = generate_sens_matrices(im, s_in, n_total)
  w_syns, E_syns = generate_syns_matrices(sm, n_total)

  matrices = Dict(
    :w_sens => w_sens,
    :E_sens => E_sens,
    :w_syns => w_syns,
    :E_syns => E_syns,
  )

  WiringConfig(s_in, n_total, matrices, im, sm, om)
end


function Base.getproperty(m::WiringConfig{T, IM, SM, OM}, s::Symbol) where {T<:AbstractFloat, IM<:InputMapping, SM<:SynsMapping, OM<:OutputMapping}
  if s === :s_in
    return getfield(m, :s_in)
  elseif s === :n_total
    return getfield(m, :n_total)
  elseif s === :n_out
    return get_n_out(m.output_mapping, m.n_total)
  elseif s === :input_mapping
    return getfield(m, :input_mapping)
  elseif s === :output_mapping
    return getfield(m, :output_mapping)
  elseif s === :syns_mapping
    return getfield(m, :syns_mapping)
  elseif s === :matrices
    return getfield(m, :matrices)
  else
    return getfield(m, s)
  end
end

default_weight() = 1
random_polarity(p=[-1,1]) = p[rand(1:length(p))]

function add_synapse!(src, dst, w, E)
  w[src,dst] = default_weight()
  E[src,dst] = random_polarity()
end

function connect_layers!(srcs, dsts, w, E; conns=Inf, handle_unconnected=true)
  for src in srcs
    n = 0
    for dst in dsts
      n > conns && break
      add_synapse!(src, dst, w, E)
      n += 1
    end
  end

  handle_unconnected == false && return

  mean_out = Int(ceil(mean([sum(w[i,dsts]) for i in srcs])))
  no_in = Int[]
  for dst in dsts
    sum(w[srcs,dst]) == 0 && push!(no_in, dst)
  end
  for dst in no_in
    for _ in 1:mean_out
      src = rand(srcs)
      add_synapse!(src, dst, w, E)
    end
  end
end


# struct Wiring{T<:AbstractFloat}
#   s_in::Int
#   n_total::Int
#   n_in::Int
#   n_out::Int
  
#   n_sensory::Int    
#   n_inter::Int      
#   n_command::Int    
#   n_motor::Int      

#   matrices::Dict{Symbol, Matrix{T}}
# end

# Wiring(s_in, n_total, n_in, n_out, matrices; n_sensory=0, n_inter=0, n_command=0, n_motor=0) = 
#   Wiring(s_in, n_total, n_in, n_out, n_sensory, n_inter, n_command, n_motor, matrices)

# FullyConnected(s_in::Int, n_total::Int; T::DataType=Float32) = FullyConnected(s_in, n_total, n_total, n_total; T)
# function FullyConnected(s_in, n_total, n_in, n_out; T::DataType=Float32)
#   @assert n_out <= n_total
#   @assert n_in <= n_total

#   n_neurons = n_total

#   w_sens = zeros(T, s_in, n_neurons)
#   E_sens = zeros(T, s_in, n_neurons)
#   w_syns = zeros(T, n_neurons, n_neurons)
#   E_syns = zeros(T, n_neurons, n_neurons)

#   for s in 1:s_in
#     for d in 1:n_in
#       add_synapse!(s, d, w_sens, E_sens)
#     end
#   end
#   for s in 1:n_neurons
#     for d in 1:n_neurons
#       add_synapse!(s, d, w_syns, E_syns)
#     end
#   end

#   matrices = Dict(
#     :w_sens => w_sens,
#     :E_sens => E_sens,
#     :w_syns => w_syns,
#     :E_syns => E_syns,
#   )
#   Wiring(s_in, n_total, n_in, n_out, matrices)
# end



# function FullSensNCPWiring(s_in::Integer; T::DataType=Float32,
#                 n_sensory=2, n_inter=3, n_command=5, n_motor=1,
#                 sensory_inter=2, inter_command=2, command_command=2, command_motor=3)

#   FullSensNCPWiring(s_in, n_sensory, n_motor; T, n_sensory, n_inter, n_command, n_motor,
#                 sensory_inter, inter_command, command_command, command_motor)
# end
# function FullSensNCPWiring(s_in::Integer, n_in::Integer, n_out::Integer; T::DataType=Float32,
#                 n_sensory=2, n_inter=3, n_command=5, n_motor=1,
#                 sensory_inter=2, inter_command=2, command_command=2, command_motor=3)

#   n_neurons = n_sensory + n_inter + n_command + n_motor
#   n_total = n_neurons

#   r_in = 1:s_in

#   s = 1
#   r_sens = s:s+n_sensory-1
#   s += n_sensory
#   r_inter = s:s+n_inter-1
#   s += n_inter
#   r_command = s:s+n_command-1
#   s += n_command
#   r_motor = s:s+n_motor-1

#   w_sens = zeros(T, s_in, n_neurons)
#   E_sens = zeros(T, s_in, n_neurons)
#   w_syns = zeros(T, n_neurons, n_neurons)
#   E_syns = zeros(T, n_neurons, n_neurons)

#   connect_layers!(r_in, r_sens, w_sens, E_sens; handle_unconnected=false)

#   connect_layers!(r_sens,    r_inter,   w_syns, E_syns; conns=sensory_inter)
#   connect_layers!(r_inter,   r_command, w_syns, E_syns; conns=inter_command)
#   connect_layers!(r_command, r_command, w_syns, E_syns; conns=command_command, handle_unconnected=false)
#   connect_layers!(r_command, r_motor,   w_syns, E_syns; conns=command_motor)

#   matrices = Dict(
#     :w_sens => w_sens,
#     :E_sens => E_sens,
#     :w_syns => w_syns,
#     :E_syns => E_syns,
#   )
  
#   Wiring(s_in, n_total, n_in, n_out, matrices; n_sensory, n_inter, n_command, n_motor)
# end

# #w = FullSensNCPWiring(10,3)
# #heatmap(w.matrices[:w_sens])
