# WIP

fff(x) = x
ModelingToolkit.@register fff(x)

function gen_sys_ltc_syn_state(wiring; T=Float32, name)
  @variables t
  D = Differential(t)

  n_neurons = wiring.n_total
  n_in = s_in = wiring.s_in
  w_sens = wiring.matrices[:w_sens]
  w_syns = wiring.matrices[:w_syns]

  sts = @variables begin
    (v[1:n_neurons](t) = rand_uniform(T, -0.2, 0.02, n_neurons)), [lower=T(-1), upper=T(1)]
    I_sens[1:n_neurons](t)
    I_syns[1:n_neurons](t)
    I_L[1:n_neurons](t)
  end

  ps = @parameters begin
    stim[1:n_neurons] = fill(T(1), n_neurons)
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
    for s in 1:n_in
      w_sens[s,d] == 0 && continue

      ss = Symbol("a_sens$(s)$(d)")
      a_sens = @variables ($ss(t) = rand_uniform(T, 0.001, 0.1)), [lower=T(0), upper=T(1)]
      ss = Symbol("α_A_sens$(s)$(d)")
      α_A_sens = @parameters $ss = rand_uniform(T, 0.2, 1), [lower=T(0.1), upper=T(20)]
      ss = Symbol("α_vh_sens$(s)$(d)")
      α_vh_sens = @parameters $ss = rand_uniform(T, -0.2, 0.4), [lower=T(-1), upper=T(1)]
      ss = Symbol("α_k_sens$(s)$(d)")
      α_k_sens = @parameters $ss = rand_uniform(T, 0.5, 0.9), [lower=T(0.001), upper=T(1)]
      ss = Symbol("β_A_sens$(s)$(d)")
      β_A_sens = @parameters $ss = rand_uniform(T, 0.2, 1), [lower=T(0.1), upper=T(20)]
      ss = Symbol("β_vh_sens$(s)$(d)")
      β_vh_sens = @parameters $ss = rand_uniform(T, -0.2, 0.4), [lower=T(-1), upper=T(1)]
      ss = Symbol("β_k_sens$(s)$(d)")
      β_k_sens = @parameters $ss = rand_uniform(T, 0.5, 0.9), [lower=T(0.01), upper=T(10)]
      ss = Symbol("G_sens$(s)$(d)")
      G_sens = @parameters $ss = rand_uniform(T, 0.001, 0.1), [lower=T(0), upper=T(1)]
      ss = Symbol("E_sens$(s)$(d)")
      E_sens = @parameters $ss = rand_uniform(T, -0.5, 0.5), [lower=T(-1), upper=T(1)]

      I_senss[s,d] = G_sens[1] * a_sens[1] * (v[d] - E_sens[1])
      α_sens = α_A_sens[1] * exp((fff(stim[s]) - α_vh_sens[1]) / α_k_sens[1])
      β_sens = β_A_sens[1] * ((fff(stim[s]) - β_vh_sens[1]) / (exp((fff(stim[s]) - β_vh_sens[1]) / β_k_sens[1]) - 1f0))

      push!(eqs, D(a_sens[1]) ~ α_sens * (1f0 - a_sens[1]) - β_sens * a_sens[1])

      push!(sts, a_sens[1])
      push!(ps, α_A_sens[1], α_vh_sens[1], α_k_sens[1], β_A_sens[1], β_vh_sens[1], β_k_sens[1], G_sens[1], E_sens[1])

    end
    for s in 1:n_neurons
      w_syns[s,d] == 0 && continue

      ss = Symbol("a_syns$(s)$(d)")
      a_syns = @variables ($ss(t) = rand_uniform(T, 0.001, 0.1)), [lower=T(0), upper=T(1)]
      ss = Symbol("α_A_syns$(s)$(d)")
      α_A_syns = @parameters $ss = rand_uniform(T, 0.2, 1), [lower=T(0.1), upper=T(20)]
      ss = Symbol("α_vh_syns$(s)$(d)")
      α_vh_syns = @parameters $ss = rand_uniform(T, -0.2, 0.4), [lower=T(-1), upper=T(1)]
      ss = Symbol("α_k_syns$(s)$(d)")
      α_k_syns = @parameters $ss = rand_uniform(T, 0.5, 0.9), [lower=T(0.001), upper=T(1)]
      ss = Symbol("β_A_syns$(s)$(d)")
      β_A_syns = @parameters $ss = rand_uniform(T, 0.2, 1), [lower=T(0.1), upper=T(20)]
      ss = Symbol("β_vh_syns$(s)$(d)")
      β_vh_syns = @parameters $ss = rand_uniform(T, -0.2, 0.4), [lower=T(-1), upper=T(1)]
      ss = Symbol("β_k_syns$(s)$(d)")
      β_k_syns = @parameters $ss = rand_uniform(T, 0.5, 0.9), [lower=T(0.01), upper=T(10)]
      ss = Symbol("G_syns$(s)$(d)")
      G_syns = @parameters $ss = rand_uniform(T, 0.001, 0.2), [lower=T(0), upper=T(1)]
      ss = Symbol("E_syns$(s)$(d)")
      E_syns = @parameters $ss = rand_uniform(T, -0.5, 0.5), [lower=T(-1), upper=T(1)]

      I_synss[s,d] = G_syns[1] * a_syns[1] * (v[d] - E_syns[1])
      α_syns = α_A_syns[1] * exp((v[s] - α_vh_syns[1]) / α_k_syns[1])
      β_syns = β_A_syns[1] * ((v[s] - β_vh_syns[1]) / (exp((v[s] - β_vh_syns[1]) / β_k_syns[1]) - 1))

      push!(eqs, D(a_syns[1]) ~ α_syns * (1 - a_syns[1]) - β_syns * a_syns[1])

      push!(sts, a_syns[1])
      push!(ps, α_A_syns[1], α_vh_syns[1], α_k_syns[1], β_A_syns[1], β_vh_syns[1], β_k_syns[1], G_syns[1], E_syns[1])
    end
  end
  for i in 1:n_neurons
    push!(eqs, I_sens[i] ~ sum(I_senss[:,i]))
    push!(eqs, I_syns[i] ~ sum(I_synss[:,i]))
  end

  ODESystem(eqs, t, sts, ps; systems, name)
end

#Zygote.@nograd gen_sys_ltc_syn_state

function LTCSynStateMTK(wiring, solver, sensealg; 
  T=Float32, tspan=T.((0, 1)),
  #wiring=nothing,
  mtkize=false, gen_jac=false, kwargs...)

  @named sys = gen_sys_ltc_syn_state(wiring)
  ssys = structural_simplify(sys)
  defs = ModelingToolkit.get_defaults(ssys)
  prob = ODEProblem(ssys, defs, tspan, tgrad=true, jac=true)

  lb, ub = BCTRNN.get_bounds(ssys, wiring.s_in; T)

  rnncell = BCTRNNCell(wiring, solver, sensealg, prob, lb, ub; kwargs...)
  MyRecur(rnncell)
end