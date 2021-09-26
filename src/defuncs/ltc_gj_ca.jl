# WIP

@fastmath function ltc_gj_ca!(du,u,p,t, n_in, n_neurons, w_sens, w_syns, w_gj)
  dv = @view du[1:n_neurons]
  dCa = @view du[n_neurons:end]
  v = @view u[1:n_neurons]
  Ca = @view u[n_neurons:end]
  stim, C, G_L, E_L, G_sens, s_sens, h_sens, E_sens, G_syns, s_syns, h_syns, E_syns, G_gj = ltc_gj_unpack_p(n_in,n_neurons,p)
  
  I_senss = zeros(eltype(stim), n_in, n_neurons)
  I_synss = zeros(eltype(stim), n_neurons, n_neurons)
  I_gjs = zeros(eltype(stim), n_neurons, n_neurons)
  #I_tot = zeros(eltype(stim), n_neurons)
  @inbounds @simd for n in 1:n_neurons
    for s in 1:n_in
      I_senss[s,n] = w_sens[s,n] * G_sens[s,n] * mysigm((stim[s] - h_sens[s,n]) * s_sens[s,n]) * (v[n] - E_sens[s,n])
    end
    for s in 1:n_neurons
      I_synss[s,n] = w_syns[s,n] * G_syns[s,n] * mysigm((v[s] - h_syns[s,n]) * s_syns[s,n]) * (v[n] - E_syns[s,n])
      I_gjs[s,n] = w_gj[s,n] * G_gj[s,n] * (v[n] - v[s])
    end
  end
  I_sens = reshape(sum(I_senss, dims=1), :)
  I_syns = reshape(sum(I_synss, dims=1), :)
  I_gj = reshape(sum(I_gjs, dims=1), :)

  I_L = (G_L .* (v .- E_L))

  #I_Ca 

  I_tot = I_L .+ I_sens .+ I_syns .+ I_gj #.+ I_Ca

  dv .= (-(I_tot) ./ C)
  #dCa .= (-I_Ca / 2*F*vol) - β*Ca 

  nothing
end

function ltc_gj_ca_init_p(n_in, n_neurons; T=Float32)
  stim   = fill(T(0), n_in) 
  #w_sens = zeros(T, n_in, n_neurons)
  #w_syns = zeros(T, n_neurons, n_neurons)
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

  G_gj   = rand_uniform(T,  0.0001,  0.1,     n_neurons, n_neurons)

  v = rand_uniform(T, 0.01, 0.1, n_neurons)
  Ca = rand_uniform(T, 0.001, 0.01, n_neurons)

  #u0 = ComponentArray(v=v)
  #p = ComponentArrays(stim=stim, C=C, G_L=G_L, E_L=E_L, 
  #                G_sens=G_sens, s_sens=s_sens, h_sens=h_sens, E_sens=E_sens,
  #                G_syns=G_syns, s_syns=s_syns, h_syns=h_syns, E_syns=E_syns)
  u0 = vcat(v, Ca)
  p = vcat(stim, #vec(w_sens), vec(w_syns),
           C, G_L, E_L,
           vec(G_sens), vec(s_sens), vec(h_sens), vec(E_sens),
           vec(G_syns), vec(s_syns), vec(h_syns), vec(E_syns),
           vec(G_gj),)
  return u0, p
end

function ltc_gj_ca_bounds(n_in, n_neurons; T=Float32)
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
    [0 for _ in 1:n_neurons*n_neurons],       # G_gj

    [-0.4 for _ in 1:n_neurons],              # v
    [0 for _ in 1:n_neurons]),             # Ca

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
    [1 for _ in 1:n_neurons*n_neurons],       # G_gj
    
    [0.8 for _ in 1:n_neurons], # v
    [100 for _ in 1:n_neurons],) # Ca

  T.(lb), T.(ub)
end


function ltc_gj_ca_unpack_p(n_in,n_neurons,p)
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

  G_gjl = synsl
  
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

  G_gj = @view p[reshape(s:s+G_gjl-1, n_neurons, n_neurons)]
  s += G_gjl

  return stim, C, G_L, E_L, G_sens, s_sens, h_sens, E_sens, G_syns, s_syns, h_syns, E_syns, G_gj
end
