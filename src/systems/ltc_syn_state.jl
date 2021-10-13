# WIP

@fastmath function ltc_syn_state!(du,u,p,t, n_in, n_neurons, w_sens, w_syns)
  v, sensa, synsa    = ltc_syn_state_unpack_u(n_in, n_neurons, u)
  dv, dsensa, dsynsa = ltc_syn_state_unpack_u(n_in, n_neurons, du)
  stim, C, G_L, E_L, G_sens, α_sensa_A, α_sensa_vh, α_sensa_k, β_sensa_A, β_sensa_vh, β_sensa_k, E_sens, G_syns, α_synsa_A, α_synsa_vh, α_synsa_k, β_synsa_A, β_synsa_vh, β_synsa_k, E_syns = ltc_syn_state_unpack_p(n_in,n_neurons,p)
  
  I_senss = zeros(eltype(stim), n_in, n_neurons)
  I_synss = zeros(eltype(stim), n_neurons, n_neurons)
  I_L = zeros(eltype(stim), n_neurons)

  @.. I_L = G_L * (v - E_L)

  α_sensa = zeros(eltype(stim), n_in, n_neurons)
  β_sensa = zeros(eltype(stim), n_in, n_neurons)
  α_synsa = zeros(eltype(stim), n_neurons, n_neurons)
  β_synsa = zeros(eltype(stim), n_neurons, n_neurons)

  @inbounds @simd for d in 1:n_neurons
    for s in 1:n_in
      I_senss[s,d] = w_sens[s,d] * G_sens[s,d] * sensa[s,d] * (v[d] - E_sens[s,d])
      α_sensa[s,d] = α_sensa_A[s,d] * exp((stim[s]-α_sensa_vh[s,d])/α_sensa_k[s,d])
      β_sensa[s,d] = β_sensa_A[s,d] * ((stim[s] -β_sensa_vh[s,d]) / (exp((stim[s] -β_sensa_vh[s,d])/β_sensa_k[s,d]) - 1f0))
    end
    for s in 1:n_neurons
      I_synss[s,d] = w_syns[s,d] * G_syns[s,d] * synsa[s,d] * (v[d] - E_syns[s,d])
      α_synsa[s,d] = α_synsa_A[s,d] * exp((v[s]-α_synsa_vh[s,d])/α_synsa_k[s,d])
      β_synsa[s,d] = β_synsa_A[s,d] * ((v[s] -β_synsa_vh[s,d]) / (exp((v[s] -β_synsa_vh[s,d])/β_synsa_k[s,d]) - 1f0))
    end
  end
  I_sens = reshape(sum(I_senss, dims=1), :)
  I_syns = reshape(sum(I_synss, dims=1), :)

  I_tot = I_L .+ I_sens .+ I_syns
  
  @.. dv = -I_tot / C
  @.. dsensa = w_sens * (α_sensa * (1f0 - sensa) - β_sensa * sensa)
  @.. dsynsa = w_syns * (α_synsa * (1f0 - synsa) - β_synsa * synsa)

  nothing
end

function ltc_syn_state_init_u0p(n_in, n_neurons; T=Float32)
  stim   = fill(T(0), n_in)
  #w_sens = zeros(T, n_in, n_neurons)
  #w_syns = zeros(T, n_neurons, n_neurons)
  C      = rand_uniform(T,  1.0,     1.002,   n_neurons)
  G_L    = rand_uniform(T,  0.0001,  0.1,     n_neurons)
  E_L    = rand_uniform(T, -0.3,     0.3,     n_neurons) #.± 0.1f0
  G_sens = rand_uniform(T,  0.0001,  0.1,     n_in, n_neurons)
  α_sensa_A  = rand_uniform(T, 0.1, 1,     n_in, n_neurons)
  α_sensa_vh = rand_uniform(T, 0.2, 0.4, n_in, n_neurons)
  α_sensa_k  = rand_uniform(T, 0.1, 0.9,     n_in, n_neurons)
  β_sensa_A  = rand_uniform(T, 0.1, 1,     n_in, n_neurons)
  β_sensa_vh = rand_uniform(T, 0.2, 0.4, n_in, n_neurons)
  β_sensa_k  = rand_uniform(T, 0.1, 0.9,     n_in, n_neurons)
  E_sens = rand_uniform(T, -0.5,     0.5,     n_in, n_neurons)
  G_syns = rand_uniform(T,  0.0001,  0.1,     n_neurons, n_neurons)
  α_synsa_A  = rand_uniform(T, 0.1, 1,     n_neurons, n_neurons)
  α_synsa_vh = rand_uniform(T, 0.2, 0.4, n_neurons, n_neurons)
  α_synsa_k  = rand_uniform(T, 0.1, 0.9,     n_neurons, n_neurons)
  β_synsa_A  = rand_uniform(T, 0.1, 1,     n_neurons, n_neurons)
  β_synsa_vh = rand_uniform(T, 0.2, 0.4, n_neurons, n_neurons)
  β_synsa_k  = rand_uniform(T, 0.1, 0.9,     n_neurons, n_neurons)
  E_syns = rand_uniform(T, -0.5,     0.5,     n_neurons, n_neurons)
  

  v = rand_uniform(T, 0.01, 0.1, n_neurons) #.± 0.1f0
  sensa = rand_uniform(T, 0.001, 0.01, n_in, n_neurons)
  synsa = rand_uniform(T, 0.001, 0.01, n_neurons, n_neurons)

  u0 = vcat(v, vec(sensa), vec(synsa))
  p = vcat(stim,
    C, G_L, E_L,
    vec(G_sens), vec(α_sensa_A),vec(α_sensa_vh),vec(α_sensa_k),vec(β_sensa_A),vec(β_sensa_vh),vec(β_sensa_k), vec(E_sens),
    vec(G_syns), vec(α_synsa_A),vec(α_synsa_vh),vec(α_synsa_k),vec(β_synsa_A),vec(β_synsa_vh),vec(β_synsa_k), vec(E_syns),
  )
  return u0, p
end

function ltc_syn_state_bounds(n_in, n_neurons; T=Float32)
  lb = vcat(
    [1.0 for _ in 1:n_neurons],               # C
    [0 for _ in 1:n_neurons],                 # G_L
    [-1 for _ in 1:n_neurons],                # E_L
    [0 for _ in 1:n_in*n_neurons],            # G_sens
    [0.01 for _ in 1:n_in*n_neurons],            # α_sensa_A
    [0 for _ in 1:n_in*n_neurons],            # α_sensa_vh
    [0.01 for _ in 1:n_in*n_neurons],            # α_sensa_k
    [0.01 for _ in 1:n_in*n_neurons],            # β_sensa_A
    [0 for _ in 1:n_in*n_neurons],            # β_sensa_vh
    [0.01 for _ in 1:n_in*n_neurons],            # β_sensa_k
    [-1 for _ in 1:n_in*n_neurons],           # E_sens
    [0 for _ in 1:n_neurons*n_neurons],       # G_syns
    [0.01 for _ in 1:n_neurons*n_neurons],            # α_synsa_A
    [0 for _ in 1:n_neurons*n_neurons],            # α_synsa_vh
    [0.01 for _ in 1:n_neurons*n_neurons],            # α_synsa_k
    [0.01 for _ in 1:n_neurons*n_neurons],            # β_synsa_A
    [0 for _ in 1:n_neurons*n_neurons],            # β_synsa_vh
    [0.01 for _ in 1:n_neurons*n_neurons],            # β_synsa_k
    [-1 for _ in 1:n_neurons*n_neurons],      # E_syns

    [-0.4 for _ in 1:n_neurons],             # v
    [0 for _ in 1:n_in*n_neurons],            # sensa
    [0 for _ in 1:n_neurons*n_neurons])             # synsa

  ub = vcat(
    [2.0 for _ in 1:n_neurons],
    [0.4 for _ in 1:n_neurons],
    [1 for _ in 1:n_neurons],
    [1 for _ in 1:n_in*n_neurons],
    [1.1 for _ in 1:n_in*n_neurons],            # α_sensa_A
    [1 for _ in 1:n_in*n_neurons],            # α_sensa_vh
    [1 for _ in 1:n_in*n_neurons],            # α_sensa_k
    [1 for _ in 1:n_in*n_neurons],            # β_sensa_A
    [1 for _ in 1:n_in*n_neurons],            # β_sensa_vh
    [1 for _ in 1:n_in*n_neurons],            # β_sensa_k
    [1 for _ in 1:n_in*n_neurons],
    [1 for _ in 1:n_neurons*n_neurons],
    [1.1 for _ in 1:n_neurons*n_neurons],            # α_synsa_A
    [1 for _ in 1:n_neurons*n_neurons],            # α_synsa_vh
    [1 for _ in 1:n_neurons*n_neurons],            # α_synsa_k
    [1 for _ in 1:n_neurons*n_neurons],            # β_synsa_A
    [1 for _ in 1:n_neurons*n_neurons],            # β_synsa_vh
    [1 for _ in 1:n_neurons*n_neurons],            # β_synsa_k
    [1 for _ in 1:n_neurons*n_neurons],
    
    [0.2 for _ in 1:n_neurons],
    [0.1 for _ in 1:n_in*n_neurons],            # sensa
    [0.1 for _ in 1:n_neurons*n_neurons],
    
    ) # v)

  T.(lb), T.(ub)
end

function ltc_syn_state_unpack_u(n_in,n_neurons,u)
  vl = n_neurons
  sensal = n_in*n_neurons
  synsal = n_neurons*n_neurons

  s = 1
  v = @view u[s:s+vl-1]
  s += vl
  sensa = @view u[reshape(s:s+sensal-1, n_in, n_neurons)]
  s += sensal
  synsa = @view u[reshape(s:s+synsal-1, n_neurons, n_neurons)]
  return v, sensa, synsa
end


function ltc_syn_state_unpack_p(n_in,n_neurons,p)
  stiml = n_in
  Cl = n_neurons
  G_Ll = n_neurons
  E_Ll = n_neurons
  sensl = n_in*n_neurons
  synsl = n_neurons*n_neurons
  G_sensl = sensl
  α_sensa_Al = sensl
  α_sensa_vhl = sensl
  α_sensa_kl = sensl
  β_sensa_Al = sensl
  β_sensa_vhl = sensl
  β_sensa_kl = sensl
  E_sensl = sensl
  G_synsl = synsl
  α_synsa_Al = synsl
  α_synsa_vhl = synsl
  α_synsa_kl = synsl
  β_synsa_Al = synsl
  β_synsa_vhl = synsl
  β_synsa_kl = synsl
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
  α_sensa_A = @view p[reshape(s:s+α_sensa_Al-1, n_in, n_neurons)]
  s += α_sensa_Al
  α_sensa_vh = @view p[reshape(s:s+α_sensa_vhl-1, n_in, n_neurons)]
  s += α_sensa_vhl
  α_sensa_k = @view p[reshape(s:s+α_sensa_kl-1, n_in, n_neurons)]
  s += α_sensa_kl
  β_sensa_A = @view p[reshape(s:s+β_sensa_Al-1, n_in, n_neurons)]
  s += β_sensa_Al
  β_sensa_vh = @view p[reshape(s:s+β_sensa_vhl-1, n_in, n_neurons)]
  s += β_sensa_vhl
  β_sensa_k = @view p[reshape(s:s+β_sensa_kl-1, n_in, n_neurons)]
  s += β_sensa_kl
  E_sens = @view p[reshape(s:s+E_sensl-1, n_in, n_neurons)]
  s += E_sensl

  G_syns = @view p[reshape(s:s+G_synsl-1, n_neurons, n_neurons)]
  s += G_synsl
  α_synsa_A = @view p[reshape(s:s+α_synsa_Al-1, n_neurons, n_neurons)]
  s += α_sensa_Al
  α_synsa_vh = @view p[reshape(s:s+α_synsa_vhl-1, n_neurons, n_neurons)]
  s += α_synsa_vhl
  α_synsa_k = @view p[reshape(s:s+α_synsa_kl-1, n_neurons, n_neurons)]
  s += α_synsa_kl
  β_synsa_A = @view p[reshape(s:s+β_synsa_Al-1, n_neurons, n_neurons)]
  s += β_synsa_Al
  β_synsa_vh = @view p[reshape(s:s+β_synsa_vhl-1, n_neurons, n_neurons)]
  s += β_synsa_vhl
  β_synsa_k = @view p[reshape(s:s+β_synsa_kl-1, n_neurons, n_neurons)]
  s += β_synsa_kl
  E_syns = @view p[reshape(s:s+E_synsl-1, n_neurons, n_neurons)]
  s += E_synsl
  return stim, C, G_L, E_L, G_sens, α_sensa_A, α_sensa_vh, α_sensa_k, β_sensa_A, β_sensa_vh, β_sensa_k, E_sens, G_syns, α_synsa_A, α_synsa_vh, α_synsa_k, β_synsa_A, β_synsa_vh, β_synsa_k, E_syns
end





function LTCSynState(wiring, solver, sensealg; 
  T=Float32, tspan=T.((0, 1)),
  #wiring=nothing,
  mtkize=true, gen_jac=false, kwargs...)

  w_sens = wiring.matrices[:w_sens]
  w_syns = wiring.matrices[:w_syns]
  
  s_in = wiring.s_in
  n_neurons = wiring.n_total

  lb, ub = ltc_syn_state_bounds(s_in, n_neurons; T)
  u0, p = ltc_syn_state_init_u0p(s_in, n_neurons; T)


  dudt!(du,u,p,t) = ltc_syn_state!(du,u,p,t, s_in, n_neurons, w_sens, w_syns)

  rnncell = BCTRNNCell(wiring, solver, sensealg, dudt!, u0, tspan, p, lb, ub; mtkize, gen_jac, kwargs...)
  MyRecur(rnncell)
end
