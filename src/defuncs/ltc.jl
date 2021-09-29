@fastmath @views function ltc!(du,u,p,t, n_in, n_neurons, w_sens, w_syns)
  v = u
  stim, C, G_L, E_L, G_sens, s_sens, h_sens, E_sens, G_syns, s_syns, h_syns, E_syns = ltc_unpack_p(n_in,n_neurons,p)

  #I_senss = reshape([G_sens[s,n] * NNlib.σ((stim[s] - h_sens[s,n]) * s_sens[s,n]) * (v[n] - E_sens[s,n]) for s in 1:size(G_sens,1), n in 1:size(G_sens,2)], size(G_sens,1), :)
  #I_synss = reshape([G_syns[s,n] * NNlib.σ((v[s] - h_syns[s,n]) * s_syns[s,n]) * (v[n] - E_syns[s,n]) for s in 1:size(G_syns,1), n in 1:size(G_syns,2)], size(G_syns,1), :)
  
  #I_senss = Array{eltype(stim)}(undef, n_in, n_neurons)
  #I_synss = Array{eltype(stim)}(undef, n_neurons, n_neurons)
  I_senss = zeros(eltype(stim), n_in, n_neurons)
  I_synss = zeros(eltype(stim), n_neurons, n_neurons)

  # TODO: tullio?
  for n in 1:n_neurons
    for s in 1:n_in
      I_senss[s,n] = w_sens[s,n] * G_sens[s,n] * mysigm((stim[s] - h_sens[s,n]) * s_sens[s,n]) * (v[n] - E_sens[s,n])
    end
  end

  for n in 1:n_neurons
    for s in 1:n_neurons
      I_synss[s,n] = w_syns[s,n] * G_syns[s,n] * mysigm((v[s] - h_syns[s,n]) * s_syns[s,n]) * (v[n] - E_syns[s,n])
    end
  end
  I_sens = reshape(sum(I_senss, dims=1), :)
  I_syns = reshape(sum(I_synss, dims=1), :)




 

  # I_sens = zeros(eltype(stim), n_neurons)
  # I_syns = zeros(eltype(stim), n_neurons)
  # @inbounds for n in 1:n_neurons
  #   for s in 1:n_in
  #     #w_sens[s,n] == 0 && continue
  #     I_sens[n] += w_sens[s,n] * G_sens[s,n] * mysigm((stim[s] - h_sens[s,n]) * s_sens[s,n]) * (v[n] - E_sens[s,n])
  #   end
  #   # I_sens[n] += sum(G_sens[:,n] .* mysigm.((stim .- h_sens[:,n]) .* s_sens[:,n]) .* (v[n] .- E_sens[:,n]))
  #   for s in 1:n_neurons
  #     #w_sens[s,n] == 0 && continue
  #     I_syns[n] += w_syns[s,n] * G_syns[s,n] * mysigm((v[s] - h_syns[s,n]) * s_syns[s,n]) * (v[n] - E_syns[s,n])
  #   end
  #   # I_syns[n] += sum(G_syns[:,n] .* mysigm.((v .- h_syns[:,n]) .* s_syns[:,n]) .* (v[n] .- E_syns[:,n]))
  # end

  #du .= (-((G_L .* (v .- E_L)) .+ I_sens .+ I_syns) ./ C) #.* dZ
  @.. du = (-((G_L * (v - E_L)) + I_sens + I_syns) / C) #.* dZ

  nothing
end

function ltc_init_u0p(n_in, n_neurons; T=Float32, wiring=nothing)
  v = rand_uniform(T, 0.01, 0.1, n_neurons) #.± 0.1f0

  stim   = fill(T(0), n_in)
  C      = rand_uniform(T,  1.0,     1.002,   n_neurons)
  G_L    = rand_uniform(T,  0.0001,  0.1,     n_neurons)
  E_L    = rand_uniform(T, -0.3,     0.3,     n_neurons) #.± 0.1f0
  G_sens = rand_uniform(T,  0.0001,  0.1,     n_in, n_neurons)
  s_sens = rand_uniform(T,  3,       8,       n_in, n_neurons)
  h_sens = rand_uniform(T,  0.3,     0.8,     n_in, n_neurons)
  E_sens = rand_uniform(T, -0.9,     0.9,     n_in, n_neurons)
  G_syns = rand_uniform(T,  0.0001,  0.1,     n_neurons, n_neurons)
  s_syns = rand_uniform(T,  3,       8,       n_neurons, n_neurons)
  h_syns = rand_uniform(T,  0.3,     0.8,     n_neurons, n_neurons)
  E_syns = rand_uniform(T, -0.9,     0.9,     n_neurons, n_neurons)

  if wiring !== nothing
    E_sens = wiring.matrices[:E_sens]
    E_syns = wiring.matrices[:E_syns]
  end

  u0 = v
  p = vcat( stim,
    C, G_L, E_L,
    vec(G_sens), vec(s_sens), vec(h_sens), vec(E_sens),
    vec(G_syns), vec(s_syns), vec(h_syns), vec(E_syns),
  )
  return u0, p
end


function ltc_u0p_ca(n_in, n_neurons; T=Float32)
  v = rand_uniform(T, 0.01, 0.1, n_neurons) #.± 0.1f0

  stim   = fill(T(0), n_in)

  C      = rand_uniform(T,  1.0,     1.002,   n_neurons)
  G_L    = rand_uniform(T,  0.0001,  0.1,     n_neurons)
  E_L    = rand_uniform(T, -0.3,     0.3,     n_neurons) #.± 0.1f0
  G_sens = rand_uniform(T,  0.0001,  0.1,     n_in, n_neurons)
  s_sens = rand_uniform(T,  3,       8,       n_in, n_neurons)
  h_sens = rand_uniform(T,  0.3,     0.8,     n_in, n_neurons)
  E_sens = rand_uniform(T, -0.9,     0.9,     n_in, n_neurons)
  G_syns = rand_uniform(T,  0.0001,  0.1,     n_neurons, n_neurons)
  s_syns = rand_uniform(T,  3,       8,       n_neurons, n_neurons)
  h_syns = rand_uniform(T,  0.3,     0.8,     n_neurons, n_neurons)
  E_syns = rand_uniform(T, -0.9,     0.9,     n_neurons, n_neurons)

  #u0 = ComponentArray{T}(v=v)
  u0 = v
  p = ComponentArrays{T}( stim=stim, 
    C=C, G_L=G_L, E_L=E_L, 
    G_sens=G_sens, s_sens=s_sens, h_sens=h_sens, E_sens=E_sens,
    G_syns=G_syns, s_syns=s_syns, h_syns=h_syns, E_syns=E_syns)

  u0, p
end

function ltc_bounds(n_in, n_neurons; T=Float32)
  lb = vcat(
    [1.0 for _ in 1:n_neurons],               # C
    [0 for _ in 1:n_neurons],                 # G_L
    [-1 for _ in 1:n_neurons],                # E_L
    [0 for _ in 1:n_in*n_neurons],            # G_sens
    [2 for _ in 1:n_in*n_neurons],            # s_sens
    [-0.3 for _ in 1:n_in*n_neurons],          # h_sens
    [-1 for _ in 1:n_in*n_neurons],           # E_sens
    [0 for _ in 1:n_neurons*n_neurons],       # G_syns
    [2 for _ in 1:n_neurons*n_neurons],       # s_syns
    [-0.3 for _ in 1:n_neurons*n_neurons],     # h_syns
    [-1 for _ in 1:n_neurons*n_neurons],      # E_syns

    [-0.4 for _ in 1:n_neurons])             # v

  ub = vcat(
    [2.0 for _ in 1:n_neurons],
    [0.4 for _ in 1:n_neurons],
    [1 for _ in 1:n_neurons],
    [1 for _ in 1:n_in*n_neurons],
    [9 for _ in 1:n_in*n_neurons],
    [0.9 for _ in 1:n_in*n_neurons],
    [1 for _ in 1:n_in*n_neurons],
    [1 for _ in 1:n_neurons*n_neurons],
    [9 for _ in 1:n_neurons*n_neurons],
    [0.9 for _ in 1:n_neurons*n_neurons],
    [1 for _ in 1:n_neurons*n_neurons],
    
    [0.8 for _ in 1:n_neurons]) # v)

  T.(lb), T.(ub)
end


function ltc_unpack_p(n_in,n_neurons,p)
  stiml = n_in
  Cl = n_neurons
  G_Ll = n_neurons
  E_Ll = n_neurons
  sensl = n_in*n_neurons
  synsl = n_neurons*n_neurons
  G_sensl = sensl
  s_sensl = sensl
  h_sensl = sensl
  E_sensl = sensl
  G_synsl = synsl
  s_synsl = synsl
  h_synsl = synsl
  E_synsl = synsl
  
  s = 1
  stim = @view p[s:s+stiml-1]
  s += stiml
  C = @view p[s:s+Cl-1]
  s += Cl
  G_L = @view p[s:s+G_Ll-1]
  s += G_Ll
  E_L = @view p[s:s+E_Ll-1]
  s += E_Ll
  G_sens = @view p[reshape(s:s+G_sensl-1, n_in, n_neurons)]
  s += G_sensl
  s_sens = @view p[reshape(s:s+s_sensl-1, n_in, n_neurons)]
  s += s_sensl
  h_sens = @view p[reshape(s:s+h_sensl-1, n_in, n_neurons)]
  s += h_sensl
  E_sens = @view p[reshape(s:s+E_sensl-1, n_in, n_neurons)]
  s += E_sensl

  G_syns = @view p[reshape(s:s+G_synsl-1, n_neurons, n_neurons)]
  s += G_synsl
  s_syns = @view p[reshape(s:s+s_synsl-1, n_neurons, n_neurons)]
  s += s_synsl
  h_syns = @view p[reshape(s:s+h_synsl-1, n_neurons, n_neurons)]
  s += h_synsl
  E_syns = @view p[reshape(s:s+E_synsl-1, n_neurons, n_neurons)]
  s += E_synsl
  return stim, C, G_L, E_L, G_sens, s_sens, h_sens, E_sens, G_syns, s_syns, h_syns, E_syns
end

function ltc_unpack_p_ca(n_in,n_neurons,p)
end

# function LTC(n_in, n_neurons, solver, sensealg; 
#   T=Float32, tspan=T.((0, 1)), n_sens=n_neurons, n_out=n_neurons, 
#   w_sens=ones(T, n_in, n_neurons), 
#   w_syns=ones(T, n_neurons, n_neurons),
#   wiring=nothing,
#   mtkize=true, gen_jac=false, kwargs...)

#   if wiring !== nothing
#     w_sens = wiring.matrices[:w_sens]
#     w_syns = wiring.matrices[:w_syns]
#   end
  
#   lb, ub = ltc_bounds(n_in, n_neurons; T)
#   u0, p = ltc_init_u0p(n_in, n_neurons; T, wiring)
#   #u0, p = ltc_sys_u0p_ca(n_in, n_neurons; T)
#   dudt!(du,u,p,t) = ltc!(du,u,p,t, n_in, n_neurons, w_sens, w_syns)

#   rnncell = BCTRNNCell(n_in, n_sens, n_neurons, n_out, solver, sensealg, dudt!, u0, tspan, p, lb, ub; mtkize, gen_jac, kwargs...)
#   MyRecur(rnncell)
# end


function LTC(wiring, solver, sensealg; 
  T=Float32, tspan=T.((0, 1)),
  #wiring=nothing,
  mtkize=true, gen_jac=false, kwargs...)

  w_sens = wiring.matrices[:w_sens]
  w_syns = wiring.matrices[:w_syns]
  
  s_in = wiring.s_in
  n_neurons = wiring.n_total

  lb, ub = ltc_bounds(s_in, n_neurons; T)
  u0, p = ltc_init_u0p(s_in, n_neurons; T, wiring)
  #u0, p = ltc_sys_u0p_ca(s_in, n_neurons; T)
  dudt!(du,u,p,t) = ltc!(du,u,p,t, s_in, n_neurons, w_sens, w_syns)

  rnncell = BCTRNNCell(wiring, solver, sensealg, dudt!, u0, tspan, p, lb, ub; mtkize, gen_jac, kwargs...)
  MyRecur(rnncell)
end




# function bcde_sys!(du,u,p,t, c_mapperp_re)
#   v, c = u
#   stim, c_mapperp_p, C, G_L, E_L, G_sens, s_sens, h_sens, E_sens, G_syns, s_syns, h_syns, E_syns = p
  
#   I_leak = G_L .* (v .- E_L)
#   I_senss = zeros(eltype(v), n_stim, size(v,1))
#   I_synss = zeros(eltype(v), size(v,1), size(v,1))

#   for n in 1:size(G_sens,2)
#     for s in 1:size(G_sens,1)
#       I_senss[s,n] = G_sens[s,n] * NNlib.σ((stim[s] - h_sens[s,n]) * s_sens[s,n]) * (v[n] - E_sens[s,n])
#     end
#   end
#   for n in 1:size(G_syns,2)
#     for s in 1:size(G_syns,1)
#       I_synss[s,n] = G_syns[s,n] * NNlib.σ((v[s] - h_syns[s,n]) * s_syns[s,n]) * (v[n] - E_syns[s,n])
#     end
#   end
#   I_sens = vec(sum(I_senss, dims=1))
#   I_syns = vec(sum(I_synss, dims=1))

#   I_tot = I_leak .+ I_sens .+ I_syns


#   du[1:size(v,1)] .= (-(I_tot .+ c) ./ C) #.* dZ

#   cmin = [vcat(stim, c[i], v[i], I_tot[i], t) for i in 1:size(v, 1)]
#   du[size(v,1)+1:2*size(v,1)] .= c_mapperp_re(c_mapperp_p)(cmin...) #.* dZ

#   nothing
# end
