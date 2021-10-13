function gen_sys_ltc_mtk(wiring; T=Float32, name)
  @variables t
  D = Differential(t)

  n_neurons = wiring.n_total
  n_in = s_in = wiring.s_in
  w_sens = wiring.matrices[:w_sens]
  w_syns = wiring.matrices[:w_syns]

  sts = @variables begin
    (v[1:n_neurons](t) = rand_uniform(T, -0.3, 0.3, n_neurons)), [lower=T(-1), upper=T(1)]
    I_sens[1:n_neurons](t)
    I_syns[1:n_neurons](t)
    I_L[1:n_neurons](t)
  end

  ps = @parameters begin
    stim[1:n_in] = fill(T(1), n_in)
    C[1:n_neurons] = rand_uniform(T, 1, 1.01, n_neurons), [lower=T(0.9), upper=T(1.1)]
    G_L[1:n_neurons] = rand_uniform(T, 0.001, 0.2, n_neurons), [lower=T(0.001), upper=T(1)]
    E_L[1:n_neurons] = rand_uniform(T, -0.5, 0.5, n_neurons), [lower=T(-1), upper=T(1)]
  end

  sts = reduce(vcat, collect.(sts))
  ps = reduce(vcat, collect.(ps))

  eqs = Equation[]
  for i in 1:n_neurons
    push!(eqs, I_L[i] ~ G_L[i] * (v[i] - E_L[i]))
    push!(eqs, D(v[i]) ~ -(I_L[i] + I_sens[i] + I_syns[i]) / C[i])
  end

  systems = ODESystem[]

  I_senss = reshape(Num[0 for _ in 1:s_in*n_neurons], s_in, n_neurons)
  I_synss = reshape(Num[0 for _ in 1:n_neurons*n_neurons], n_neurons, n_neurons)

  for d in 1:n_neurons
    
    n_synapses = Int(sum(w_sens[:,d]))
    weights_exc_sens = Num[]
    weights_inh_sens = Num[]
    
    ss = Symbol("G_exc_sens$(d)")
    G_exc_sens = @parameters $ss = rand_uniform(T, 0.001, 0.2), [lower=T(0.001), upper=T(1)]
    push!(ps, G_exc_sens[1])
    ss = Symbol("G_inh_sens$(d)")
    G_inh_sens = @parameters $ss = rand_uniform(T, 0.001, 0.2), [lower=T(0.001), upper=T(1)]
    push!(ps, G_inh_sens[1])

    ss = Symbol("sum_weights_exc_sens$(d)")
    sum_weights_exc_sens = @variables $ss(t)
    push!(sts, sum_weights_exc_sens[1])
    ss = Symbol("sum_weights_inh_sens$(d)")
    sum_weights_inh_sens = @variables $ss(t)
    push!(sts, sum_weights_inh_sens[1])

    for s in 1:n_in
      w_sens[s,d] == 0 && continue

      E_def = wiring.matrices[:E_sens][s,d]

      ss = Symbol("weight_sensp$(s)_$(d)")
      weight_sensp = @parameters $ss = T(1/n_synapses), [lower=T(0.0001), upper=T(1)]
      ss = Symbol("weight_sens$(s)_$(d)")
      weight_sens = @variables $ss(t)

      push!(eqs, weight_sens[1] ~ weight_sensp[1])

      g_sens = 0
      
      if E_def < 0
        push!(weights_exc_sens, weight_sens[1])
        g_sens = (weight_sens[1] / sum_weights_exc_sens[1]) * G_exc_sens[1]
      else
        push!(weights_inh_sens, weight_sens[1])
        g_sens = (weight_sens[1] / sum_weights_inh_sens[1]) * G_inh_sens[1]
      end
      #g_sens = weight_sens[1] * G_sens[1]
      
      ss = Symbol("vh_sens$(s)_$(d)")
      vh_sens = @parameters $ss = rand_uniform(T, -0.4, -0.2) * E_def, [lower=T(-2), upper=T(2)]
      ss = Symbol("k_sens$(s)_$(d)")
      k_sens = @parameters $ss = rand_uniform(T, 2, 4), [lower=T(-20), upper=T(20)]
      

      ss = Symbol("E_sens$(s)_$(d)")
      #E_sens = @parameters $ss = rand_uniform(T, -0.5, 0.5), [lower=T(-1), upper=T(1)]
      E = @parameters $ss = E_def, [lower=T(-1), upper=T(1)]

      I_senss[s,d] = g_sens * mysigm(k_sens[1] * (fff(stim[s]) - vh_sens[1])) * (v[d] - E[1])
     
      push!(sts, weight_sens[1])
      push!(ps, vh_sens[1], k_sens[1], weight_sensp[1], E[1])

    end
    #push!(eqs, 0 ~ sum(ws->ws, weights_sens) - 1)
    push!(eqs, sum_weights_exc_sens[1] ~ sum(weights_exc_sens))
    push!(eqs, sum_weights_inh_sens[1] ~ sum(weights_inh_sens))

    
    n_synapses = Int(sum(w_syns[:,d]))
    weights_exc_syns = Num[]
    weights_inh_syns = Num[]
    
    ss = Symbol("G_exc_syns$(d)")
    G_exc_syns = @parameters $ss = rand_uniform(T, 0.001, 0.2), [lower=T(0.001), upper=T(1)]
    push!(ps, G_exc_syns[1])
    ss = Symbol("G_inh_syns$(d)")
    G_inh_syns = @parameters $ss = rand_uniform(T, 0.001, 0.2), [lower=T(0.001), upper=T(1)]
    push!(ps, G_inh_syns[1])

    ss = Symbol("sum_weights_exc_syns$(d)")
    sum_weights_exc_syns = @variables $ss(t)
    push!(sts, sum_weights_exc_syns[1])
    ss = Symbol("sum_weights_inh_syns$(d)")
    sum_weights_inh_syns = @variables $ss(t)
    push!(sts, sum_weights_inh_syns[1])

    for s in 1:n_neurons
      w_syns[s,d] == 0 && continue

      E_def = wiring.matrices[:E_syns][s,d]

      ss = Symbol("weight_synsp$(s)_$(d)")
      weight_synsp = @parameters $ss = T(1/n_synapses), [lower=T(0.0001), upper=T(1)]
      ss = Symbol("weight_syns$(s)_$(d)")
      weight_syns = @variables $ss(t)

      push!(eqs, weight_syns[1] ~ weight_synsp[1])

      g_syns = 0
      
      if E_def < 0
        push!(weights_exc_syns, weight_syns[1])
        g_syns = (weight_syns[1] / sum_weights_exc_syns[1]) * G_exc_syns[1]
      else
        push!(weights_inh_syns, weight_syns[1])
        g_syns = (weight_syns[1] / sum_weights_inh_syns[1]) * G_inh_syns[1]
      end
      #g_syns = weight_syns[1] * G_syns[1]
      
      ss = Symbol("vh_syns$(s)_$(d)")
      vh_syns = @parameters $ss = rand_uniform(T, -0.4, -0.2) * E_def, [lower=T(-2), upper=T(2)]
      ss = Symbol("k_syns$(s)_$(d)")
      k_syns = @parameters $ss = rand_uniform(T, 2, 4), [lower=T(-20), upper=T(20)]
      

      ss = Symbol("E_syns$(s)_$(d)")
      #E_syns = @parameters $ss = rand_uniform(T, -0.5, 0.5), [lower=T(-1), upper=T(1)]
      E = @parameters $ss = E_def, [lower=T(-1), upper=T(1)]

      I_synss[s,d] = g_syns * mysigm(k_syns[1] * (v[s] - vh_syns[1])) * (v[d] - E[1])
     
      push!(sts, weight_syns[1])
      push!(ps, vh_syns[1], k_syns[1], weight_synsp[1], E[1])
    end
    #push!(eqs, 0 ~ sum(ws->ws, weights_syns) - 1)
    push!(eqs, sum_weights_exc_syns[1] ~ sum(weights_exc_syns))
    push!(eqs, sum_weights_inh_syns[1] ~ sum(weights_inh_syns))
  end

  for i in 1:n_neurons
    push!(eqs, I_sens[i] ~ sum(I_senss[:,i]))
  end
  for i in 1:n_neurons
    push!(eqs, I_syns[i] ~ sum(I_synss[:,i]))
  end

  ODESystem(eqs, t, sts, ps; systems, name)
end

#Zygote.@nograd gen_sys_ltc_syn_state

function LTCMTK(wiring, solver, sensealg; 
  T=Float32, tspan=T.((0, 1)),
  #wiring=nothing,
  tgrad=true, gen_jac=false, kwargs...)

  @named sys = gen_sys_ltc_mtk(wiring)
  ssys = structural_simplify(sys)
  defs = ModelingToolkit.get_defaults(ssys)

  prob = ODEProblem(ssys, defs, tspan, tgrad=tgrad, jac=gen_jac)

  lb, ub = BCTRNN.get_bounds(ssys, wiring.s_in; T)

  rnncell = BCTRNNCell(wiring, solver, sensealg, prob, lb, ub; kwargs...)
  MyRecur(rnncell), defs, ssys
end