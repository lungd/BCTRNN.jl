function gen_sys_dLeak(wiring; T=Float32, name)
  @variables t
  D = Differential(t)

  n_neurons = wiring.n_total
  n_in = s_in = wiring.s_in
  w_sens = wiring.matrices[:w_sens]
  w_syns = wiring.matrices[:w_syns]

  sts = @variables begin
    (v[1:n_neurons](t) = rand_uniform(T, -0.5, 0.5, n_neurons)), [lower=T(-1), upper=T(1)]
    I_sens[1:n_neurons](t)
    I_syns[1:n_neurons](t)
    I_LE[1:n_neurons](t)
    I_LI[1:n_neurons](t)
  end

  #Es_L = [random_polarity([T(-1),T(1)]) for _ in 1:n_neurons]
  Es_LE = [random_polarity([T(0.5),T(1)]) for _ in 1:n_neurons]
  Es_LI = [random_polarity([T(-0.5),T(-1)]) for _ in 1:n_neurons]

  ps = @parameters begin
    stim[1:s_in] = fill(T(1), s_in)
    C[1:n_neurons] = rand_uniform(T, 1, 1.01, n_neurons), [lower=T(0.9), upper=T(1.1)]
    G_LE[1:n_neurons] = rand_uniform(T, 0.001, 0.2, n_neurons), [lower=T(0.001), upper=T(1)]
    G_LI[1:n_neurons] = rand_uniform(T, 0.001, 0.2, n_neurons), [lower=T(0.001), upper=T(1)]
    E_LE[1:n_neurons] = Es_LE, [lower=T(0), upper=T(1)]
    E_LI[1:n_neurons] = Es_LI, [lower=T(-1), upper=T(0)]
  end

  sts = reduce(vcat, collect.(sts))
  ps = reduce(vcat, collect.(ps))

  eqs = Equation[]
  for i in 1:n_neurons
    push!(eqs, I_LE[i] ~ G_LE[i] * (v[i] - E_LE[i]))
    push!(eqs, I_LI[i] ~ G_LI[i] * (v[i] - E_LI[i]))
    push!(eqs, D(v[i]) ~ -(I_LE[i] + I_LI[i] + I_sens[i] + I_syns[i]) / C[i])
  end

  systems = ODESystem[]

  I_senss = reshape(Num[0 for _ in 1:s_in*n_neurons], s_in, n_neurons)
  I_synss = reshape(Num[0 for _ in 1:n_neurons*n_neurons], n_neurons, n_neurons)

  for d in 1:n_neurons
    for s in 1:n_in
      w_sens[s,d] == 0 && continue

      ss = Symbol("vh_sens$(s)$(d)")
      vh_sens = @parameters $ss = rand_uniform(T, -0.2, 0.4), [lower=T(-1), upper=T(1)]
      ss = Symbol("k_sens$(s)$(d)")
      k_sens = @parameters $ss = rand_uniform(T, -5, 5), [lower=T(-20), upper=T(20)]
      
      ss = Symbol("G_sens$(s)$(d)")
      G_sens = @parameters $ss = rand_uniform(T, 0.001, 0.1), [lower=T(0), upper=T(1)]
      ss = Symbol("E_sens$(s)$(d)")
      #E_sens = @parameters $ss = rand_uniform(T, -0.5, 0.5), [lower=T(-1), upper=T(1)]
      E_sens = @parameters $ss = wiring.matrices[:E_sens][s,d], [lower=T(-1), upper=T(1)]

      I_senss[s,d] = G_sens[1] * mysigm(k_sens[1] * (fff(stim[s]) - vh_sens[1])) * (v[d] - E_sens[1])
     
      push!(ps, vh_sens[1], k_sens[1], G_sens[1], E_sens[1])

    end
    for s in 1:n_neurons
      w_syns[s,d] == 0 && continue

      ss = Symbol("vh_syns$(s)$(d)")
      vh_syns = @parameters $ss = rand_uniform(T, -0.2, 0.4), [lower=T(-1), upper=T(1)]
      ss = Symbol("k_syns$(s)$(d)")
      k_syns = @parameters $ss = rand_uniform(T, -5, 5), [lower=T(-20), upper=T(20)]
      
      ss = Symbol("G_syns$(s)$(d)")
      G_syns = @parameters $ss = rand_uniform(T, 0.001, 0.1), [lower=T(0), upper=T(1)]
      ss = Symbol("E_syns$(s)$(d)")
      E_syns = @parameters $ss = wiring.matrices[:E_syns][s,d], [lower=T(-1), upper=T(1)]

      I_synss[s,d] = G_syns[1] * mysigm(k_syns[1] * (v[s] - vh_syns[1])) * (v[d] - E_syns[1])
     
      push!(ps, vh_syns[1], k_syns[1], G_syns[1], E_syns[1])
    end
  end
  for i in 1:n_neurons
    push!(eqs, I_sens[i] ~ sum(I_senss[:,i]))
    push!(eqs, I_syns[i] ~ sum(I_synss[:,i]))
  end

  ODESystem(eqs, t, sts, ps; systems, name)
end

#Zygote.@nograd gen_sys_ltc_syn_state

function DLeak(wiring, solver, sensealg; 
  T=Float32, tspan=T.((0, 1)),
  #wiring=nothing,
  tgrad=true, gen_jac=false, kwargs...)

  @named sys = gen_sys_dLeak(wiring)
  ssys = structural_simplify(sys)
  defs = ModelingToolkit.get_defaults(ssys)

  prob = ODEProblem(ssys, defs, tspan, tgrad=tgrad, jac=gen_jac)

  lb, ub = BCTRNN.get_bounds(ssys, wiring.s_in; T)

  rnncell = BCTRNNCell(wiring, solver, sensealg, prob, lb, ub; kwargs...)
  MyRecur(rnncell), defs, ssys
end