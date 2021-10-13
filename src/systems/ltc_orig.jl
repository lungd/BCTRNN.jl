function gen_sys_ltc_orig_mtk(wiring; T=Float32, name)
  @variables t
  D = Differential(t)

  ϵ = T(1e-5)

  n_neurons = wiring.n_total
  n_in = s_in = wiring.s_in
  w_sens = wiring.matrices[:w_sens]
  w_syns = wiring.matrices[:w_syns]

  sts = @variables begin
    (v[1:n_neurons](t) = rand_uniform(T, -0.2, 0.2, n_neurons))#, [lower=T(-1), upper=T(1)]
    I_sens[1:n_neurons](t)
    I_syns[1:n_neurons](t)
    I_L[1:n_neurons](t)
  end

  ps = @parameters begin
    stim[1:n_in] = fill(T(1), n_in)
    C[1:n_neurons] = rand_uniform(T, 0.4, 0.6, n_neurons), [lower=T(0), upper=T(Inf)]
    G_L[1:n_neurons] = rand_uniform(T, 0.001, 1, n_neurons), [lower=T(0), upper=T(Inf)]
    E_L[1:n_neurons] = rand_uniform(T, -0.2, 0.2, n_neurons)#, [lower=T(-1), upper=T(1)]
  end

  sts = reduce(vcat, collect.(sts))
  ps = reduce(vcat, collect.(ps))

  eqs = Equation[]
  for i in 1:n_neurons
    push!(eqs, I_L[i] ~ G_L[i] * (v[i] - E_L[i]))
    push!(eqs, D(v[i]) ~ -(I_L[i] + I_sens[i] + I_syns[i]) / (C[i] + ϵ))
  end

  systems = ODESystem[]

  I_senss = reshape(Num[0 for _ in 1:s_in*n_neurons], s_in, n_neurons)
  I_synss = reshape(Num[0 for _ in 1:n_neurons*n_neurons], n_neurons, n_neurons)

  for d in 1:n_neurons
    for s in 1:n_in
      w_sens[s,d] == 0 && continue

      ss = Symbol("G_sens$(d)")
      G = @parameters $ss = rand_uniform(T, 0.001, 1), [lower=T(0), upper=T(Inf)]
      ss = Symbol("E_sens$(d)")
      E = @parameters $ss = wiring.matrices[:E_sens][s,d]#, [lower=T(-1), upper=T(1)]
      ss = Symbol("vh_sens$(s)_$(d)")
      vh = @parameters $ss = rand_uniform(T, 0.3, 0.8)#, [lower=T(0), upper=T(1)]
      ss = Symbol("k_sens$(s)_$(d)")
      k = @parameters $ss = rand_uniform(T, 3, 8)#, [lower=T(1), upper=T(10)]

      I_senss[s,d] = G[1] * mysigm(k[1] * (fff(stim[s]) - vh[1])) * (v[d] - E[1])
     
      push!(ps, G[1], E[1], vh[1], k[1])
    end

    for s in 1:n_neurons
      w_syns[s,d] == 0 && continue

      ss = Symbol("G_syns$(d)")
      G = @parameters $ss = rand_uniform(T, 0.001, 1), [lower=T(0), upper=T(Inf)]
      ss = Symbol("E_syns$(d)")
      E = @parameters $ss = wiring.matrices[:E_sens][s,d]#, [lower=T(-1), upper=T(1)]
      ss = Symbol("vh_syns$(s)_$(d)")
      vh = @parameters $ss = rand_uniform(T, 0.3, 0.8)#, [lower=T(0), upper=T(1)]
      ss = Symbol("k_syns$(s)_$(d)")
      k = @parameters $ss = rand_uniform(T, 3, 8)#, [lower=T(1), upper=T(10)]

      I_senss[s,d] = G[1] * mysigm(k[1] * (v[s] - vh[1])) * (v[d] - E[1])
     
      push!(ps, G[1], E[1], vh[1], k[1])
    end
  end

  for i in 1:n_neurons
    push!(eqs, I_sens[i] ~ sum(I_senss[:,i]))
    push!(eqs, I_syns[i] ~ sum(I_synss[:,i]))
  end

  ODESystem(eqs, t, sts, ps; systems, name)
end

#Zygote.@nograd gen_sys_ltc_syn_state

function LTCOrigMTK(wiring, solver, sensealg; 
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