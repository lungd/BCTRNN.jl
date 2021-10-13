# WIP
# K/Ca/Leak channels. synapses with presyn NT release + d[NT] + postsyn ionotropic receptor 

function gen_sys_stateful(wiring; T=Float32, name)
  @variables t
  D = Differential(t)

  n_neurons = wiring.n_total
  n_in = s_in = wiring.s_in
  w_sens = wiring.matrices[:w_sens]
  w_syns = wiring.matrices[:w_syns]

  sts = @variables begin
    (v[1:n_neurons](t) = rand_uniform(T, -80, -40, n_neurons)), [lower=T(-90), upper=T(30)]
    (a_K[1:n_neurons](t) = rand_uniform(T, 0.02, 0.1, n_neurons)), [lower=T(0), upper=T(1)]
    (a_Ca[1:n_neurons](t) = rand_uniform(T, 0.02, 0.1, n_neurons)), [lower=T(0), upper=T(1)]
    (i_Ca[1:n_neurons](t) = rand_uniform(T, 0.5, 0.9, n_neurons)), [lower=T(0), upper=T(1)]

    I_L[1:n_neurons](t)

    a_K_α[1:n_neurons](t)
    a_K_β[1:n_neurons](t)
    I_K[1:n_neurons](t)

    a_Ca_α[1:n_neurons](t)
    a_Ca_β[1:n_neurons](t)
    i_Ca_α[1:n_neurons](t)
    i_Ca_β[1:n_neurons](t)
    I_Ca[1:n_neurons](t)
    
    I_sens[1:n_neurons](t)
    I_syns[1:n_neurons](t)
  end

  ps = @parameters begin
    stim[1:n_neurons] = fill(T(1), n_neurons)
    C[1:n_neurons] = rand_uniform(T, 1, 1.01, n_neurons), [lower=T(0.5), upper=T(2.1)]
    G_L[1:n_neurons] = rand_uniform(T, 0.001, 0.2, n_neurons), [lower=T(0.001), upper=T(1)]
    E_L[1:n_neurons] = rand_uniform(T, -80, -50, n_neurons), [lower=T(-90), upper=T(-20)]
    
    a_K_α_A[1:n_neurons] = rand_uniform(T, 0.01, 0.1, n_neurons), [lower=T(0.001), upper=T(1)] 
    a_K_α_vh[1:n_neurons] = rand_uniform(T, -40, -30, n_neurons), [lower=T(-80), upper=T(-20)]
    a_K_α_k[1:n_neurons] = rand_uniform(T, 10, 30, n_neurons), [lower=T(1), upper=T(100)]
    a_K_β_A[1:n_neurons] = rand_uniform(T, 2, 9, n_neurons), [lower=T(1), upper=T(10)] 
    a_K_β_vh[1:n_neurons] = rand_uniform(T, -50, -40, n_neurons), [lower=T(-80), upper=T(-30)]
    a_K_β_k[1:n_neurons] = rand_uniform(T, 20, 40, n_neurons), [lower=T(10), upper=T(100)]

    a_Ca_α_A[1:n_neurons] = rand_uniform(T, 0.01, 0.1, n_neurons), [lower=T(0.001), upper=T(1)] 
    a_Ca_α_vh[1:n_neurons] = rand_uniform(T, -70, -60, n_neurons), [lower=T(-80), upper=T(80)]
    a_Ca_α_k[1:n_neurons] = rand_uniform(T, 2, 20, n_neurons), [lower=T(1), upper=T(40)]
    a_Ca_β_A[1:n_neurons] = rand_uniform(T, 1, 10, n_neurons), [lower=T(0.5), upper=T(20)] 
    a_Ca_β_vh[1:n_neurons] = rand_uniform(T, -75, -50, n_neurons), [lower=T(-80), upper=T(80)]
    a_Ca_β_k[1:n_neurons] = rand_uniform(T, 10, 30, n_neurons), [lower=T(1), upper=T(100)]

    i_Ca_α_A[1:n_neurons] = rand_uniform(T, 0.01, 0.1, n_neurons), [lower=T(0.001), upper=T(1)] 
    i_Ca_α_vh[1:n_neurons] = rand_uniform(T, -40, -30, n_neurons), [lower=T(-80), upper=T(30)]
    i_Ca_α_k[1:n_neurons] = rand_uniform(T, 10, 30, n_neurons), [lower=T(1), upper=T(100)]
    i_Ca_β_A[1:n_neurons] = rand_uniform(T, 2, 9, n_neurons), [lower=T(1), upper=T(10)] 
    i_Ca_β_vh[1:n_neurons] = rand_uniform(T, -60, -50, n_neurons), [lower=T(-70), upper=T(-40)]
    i_Ca_β_k[1:n_neurons] = rand_uniform(T, 20, 40, n_neurons), [lower=T(10), upper=T(100)]

    G_K[1:n_neurons] = rand_uniform(T, 0.001, 0.2, n_neurons), [lower=T(0.001), upper=T(1)]
    E_K[1:n_neurons] = rand_uniform(T, -80, -50, n_neurons), [lower=T(-90), upper=T(-20)]
    G_Ca[1:n_neurons] = rand_uniform(T, 0.001, 0.2, n_neurons), [lower=T(0.001), upper=T(1)]
    E_Ca[1:n_neurons] = rand_uniform(T, 20, 60, n_neurons), [lower=T(10), upper=T(80)]
  end

  sts = reduce(vcat, collect.(sts))
  ps = reduce(vcat, collect.(ps))

  a_K_αf(i) = a_K_α_A[i] * ((v[i] - a_K_α_vh[i]) / (1f0 - exp(-(v[i] - a_K_α_vh[i]) / a_K_α_k[i])))
  a_K_βf(i) = a_K_β_A[i] * exp(-(v[i] - a_K_β_vh[i]) / a_K_β_k[i])

  a_Ca_αf(i) = a_Ca_α_A[i] * ((v[i] - a_Ca_α_vh[i]) / (1f0 - exp(-(v[i] - a_Ca_α_vh[i]) / a_Ca_α_k[i])))
  a_Ca_βf(i) = a_Ca_β_A[i] * exp(-(v[i] - a_Ca_β_vh[i]) / a_Ca_β_k[i])
  i_Ca_αf(i) = i_Ca_α_A[i] * exp(-(v[i] - i_Ca_α_vh[i]) / i_Ca_α_k[i])
  i_Ca_βf(i) = i_Ca_β_A[i] * ((v[i] - i_Ca_β_vh[i]) / (1f0 - exp(-(v[i] - i_Ca_β_vh[i]) / i_Ca_β_k[i])))

  eqs = Equation[]
  for i in 1:n_neurons
    push!(eqs, D(v[i]) ~ -(I_L[i] + I_K[i] + I_Ca[i] + I_sens[i] + I_syns[i]) / C[i])
  end
  for i in 1:n_neurons
    push!(eqs, I_L[i] ~ G_L[i] * (v[i] - E_L[i]))
    
    push!(eqs, I_K[i] ~ G_K[i] * a_K[i]^4 * (v[i] - E_K[i]))
    push!(eqs, a_K_α[i] ~ a_K_αf(i))
    push!(eqs, a_K_β[i] ~ a_K_βf(i))
    push!(eqs, D(a_K[i]) ~ a_K_α[i] * (1f0 - a_K[i]) - a_K_β[i]*a_K[i])
    

    push!(eqs, I_Ca[i] ~ G_Ca[i] * a_Ca[i]^3 * i_Ca[i] * (v[i] - E_Ca[i]))
    push!(eqs, a_Ca_α[i] ~ a_Ca_αf(i))
    push!(eqs, a_Ca_β[i] ~ a_Ca_βf(i))
    push!(eqs, D(a_Ca[i]) ~ a_Ca_α[i] * (1f0 - a_Ca[i]) - a_Ca_β[i]*a_Ca[i])

    push!(eqs, i_Ca_α[i] ~ i_Ca_αf(i))
    push!(eqs, i_Ca_β[i] ~ i_Ca_βf(i))
    push!(eqs, D(i_Ca[i]) ~ i_Ca_α[i] * (1f0 - i_Ca[i]) - i_Ca_β[i]*i_Ca[i])
  end

  systems = ODESystem[]

  I_senss = reshape(Num[0 for _ in 1:s_in*n_neurons], s_in, n_neurons)
  I_synss = reshape(Num[0 for _ in 1:n_neurons*n_neurons], n_neurons, n_neurons)

  for d in 1:n_neurons
    for s in 1:n_in
      w_sens[s,d] == 0 && continue

      ss = Symbol("a_nt_release_sens$(s)$(d)")
      a_nt_release_sens = @variables $ss(t)
      ss = Symbol("nt_sens$(s)$(d)")
      nt_sens = @variables ($ss(t) = rand_uniform(T, 0.001, 0.1)), [lower=T(0), upper=T(10)]
      ss = Symbol("a_r_sens$(s)$(d)")
      a_r_sens = @variables $ss(t)

      ss = Symbol("a_nt_release_sens_A$(s)$(d)")
      a_nt_release_sens_A = @parameters $ss = rand_uniform(T, 0.001, 0.9), [lower=T(0.0001), upper=T(2)]
      ss = Symbol("a_nt_release_sens_vh$(s)$(d)")
      a_nt_release_sens_vh = @parameters $ss = rand_uniform(T, -40, 20), [lower=T(-80), upper=T(80)]
      ss = Symbol("a_nt_release_sens_k$(s)$(d)")
      a_nt_release_sens_k = @parameters $ss = rand_uniform(T, 0.1, 1), [lower=T(0.05), upper=T(10)]
      
      ss = Symbol("nt_sens_t_accum$(s)$(d)")
      nt_sens_t_accum = @parameters $ss = rand_uniform(T, 0.01, 0.1), [lower=T(0.0001), upper=T(10)]
      ss = Symbol("nt_sens_t_elim$(s)$(d)")
      nt_sens_t_elim = @parameters $ss = rand_uniform(T, 0.1, 1), [lower=T(0.0001), upper=T(10)]

      ss = Symbol("rit_sens$(s)$(d)")
      rit_sens = @parameters $ss = rand_uniform(T, 0.99, 0.999), [lower=T(0.1), upper=T(1)]
      ss = Symbol("cit_sens$(s)$(d)")
      cit_sens = @parameters $ss = rand_uniform(T, 0.5, 0.9), [lower=T(0.1), upper=T(1)]


      ss = Symbol("G_sens$(s)$(d)")
      G_sens = @parameters $ss = rand_uniform(T, 0.001, 0.1), [lower=T(0), upper=T(1)]
      ss = Symbol("E_sens$(s)$(d)")
      E_sens = @parameters $ss = wiring.matrices[:E_sens][s,d], [lower=T(-80), upper=T(80)]


      push!(eqs, a_nt_release_sens[1] ~ a_nt_release_sens_A[1] / (1 + exp(-a_nt_release_sens_k[1] * (fff(stim[s]) - a_nt_release_sens_vh[1]))))
      push!(eqs, D(nt_sens[1]) ~ a_nt_release_sens[1] / nt_sens_t_accum[1] - nt_sens[1] / nt_sens_t_elim[1])
      push!(eqs, a_r_sens[1] ~ rit_sens[1] * nt_sens[1] / (cit_sens[1] + nt_sens[1]))

      I_senss[s,d] = G_sens[1] * a_r_sens[1] * (v[d] - E_sens[1])

      push!(sts, a_nt_release_sens[1], nt_sens[1], a_r_sens[1])
      push!(ps, a_nt_release_sens_A[1], a_nt_release_sens_vh[1], a_nt_release_sens_k[1], nt_sens_t_accum[1], nt_sens_t_elim[1], rit_sens[1], cit_sens[1], G_sens[1], E_sens[1])
    end
    for s in 1:n_neurons
      w_syns[s,d] == 0 && continue

      ss = Symbol("a_nt_release_syns$(s)$(d)")
      a_nt_release_syns = @variables $ss(t)
      ss = Symbol("nt_syns$(s)$(d)")
      nt_syns = @variables ($ss(t) = rand_uniform(T, 0.001, 0.1)), [lower=T(0), upper=T(10)]
      ss = Symbol("a_r_syns$(s)$(d)")
      a_r_syns = @variables $ss(t)
      

      ss = Symbol("a_nt_release_syns_A$(s)$(d)")
      a_nt_release_syns_A = @parameters $ss = rand_uniform(T, 1.01, 10), [lower=T(1), upper=T(1000)]
      ss = Symbol("a_nt_release_syns_vh$(s)$(d)")
      a_nt_release_syns_vh = @parameters $ss = rand_uniform(T, -50, 50), [lower=T(-80), upper=T(80)]
      ss = Symbol("a_nt_release_syns_k$(s)$(d)")
      a_nt_release_syns_k = @parameters $ss = rand_uniform(T, 0.1, 0.8), [lower=T(0.05), upper=T(1)]
      
      ss = Symbol("nt_syns_t_accum$(s)$(d)")
      nt_syns_t_accum = @parameters $ss = rand_uniform(T, 1, 10), [lower=T(0.0001), upper=T(10)]
      ss = Symbol("nt_syns_t_elim$(s)$(d)")
      nt_syns_t_elim = @parameters $ss = rand_uniform(T, 1, 10), [lower=T(0.0001), upper=T(10)]

      ss = Symbol("rit_syns$(s)$(d)")
      rit_syns = @parameters $ss = rand_uniform(T, 0.99, 0.999), [lower=T(0.1), upper=T(1)]
      ss = Symbol("cit_syns$(s)$(d)")
      cit_syns = @parameters $ss = rand_uniform(T, 0.5, 0.9), [lower=T(0.1), upper=T(1)]


      ss = Symbol("G_syns$(s)$(d)")
      G_syns = @parameters $ss = rand_uniform(T, 0.001, 0.1), [lower=T(0), upper=T(1)]
      ss = Symbol("E_syns$(s)$(d)")
      E_syns = @parameters $ss = wiring.matrices[:E_syns][s,d], [lower=T(-80), upper=T(80)]


      push!(eqs, a_nt_release_syns[1] ~ a_nt_release_syns_A[1] / (1 + exp(-a_nt_release_syns_k[1] * (v[s] - a_nt_release_syns_vh[1]))))
      push!(eqs, D(nt_syns[1]) ~ a_nt_release_syns[1] / nt_syns_t_accum[1] - nt_syns[1] / nt_syns_t_elim[1])
      push!(eqs, a_r_syns[1] ~ rit_syns[1] * nt_syns[1] / (cit_syns[1] + nt_syns[1]))

      I_synss[s,d] = G_syns[1] * a_r_syns[1] * (v[d] - E_syns[1])

      push!(sts, a_nt_release_syns[1], nt_syns[1], a_r_syns[1])
      push!(ps, a_nt_release_syns_A[1], a_nt_release_syns_vh[1], a_nt_release_syns_k[1], nt_syns_t_accum[1], nt_syns_t_elim[1], rit_syns[1], cit_syns[1], G_syns[1], E_syns[1])
    end
  end
  for i in 1:n_neurons
    push!(eqs, I_sens[i] ~ sum(I_senss[:,i]))
    push!(eqs, I_syns[i] ~ sum(I_synss[:,i]))
  end

  ODESystem(eqs, t, sts, ps; systems, name)
end

#Zygote.@nograd gen_sys_stateful

function StatefulMTK(wiring, solver, sensealg; 
  T=Float32, tspan=T.((0, 1)),
  #wiring=nothing,
  mtkize=false, gen_jac=false, kwargs...)

  @named sys = gen_sys_stateful(wiring)
  ssys = structural_simplify(sys)
  defs = ModelingToolkit.get_defaults(ssys)
  prob = ODEProblem(ssys, defs, tspan, tgrad=true, jac=true)

  lb, ub = BCTRNN.get_bounds(ssys, wiring.s_in; T)

  rnncell = BCTRNNCell(wiring, solver, sensealg, prob, lb, ub; kwargs...)
  MyRecur(rnncell)
end