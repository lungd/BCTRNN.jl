module BCTRNN

using Distributions
using OrdinaryDiffEq
using StochasticDiffEq
using DiffEqSensitivity
import DiffEqFlux: initial_params, paramlength, FastChain, FastDense, sciml_train, BFGS
using GalacticOptim
using Zygote
using Flux
import Flux.Data: DataLoader
using NNlib: sigmoid
using Random
using ModelingToolkit
using DataInterpolations
using Measurements
using FastBroadcast
using ComponentArrays

using Plots

#using BlackBoxOptim

import IterTools: ncycle


struct VariableLowerBound end
struct VariableUpperBound end
Symbolics.option_to_metadata_type(::Val{:lower}) = VariableLowerBound
Symbolics.option_to_metadata_type(::Val{:upper}) = VariableUpperBound

function get_bounds(sys, s_in; T=Float32, default_lb=T(-Inf), default_ub=T(Inf))
  cell_lb = T[]
  cell_ub = T[]

  params = collect(parameters(sys))[s_in+1:end]
  states = collect(ModelingToolkit.states(sys))
  for v in vcat(params,states)
    contains(string(v), "OutPin") && continue
    lower = hasmetadata(v, VariableLowerBound) ? getmetadata(v, VariableLowerBound) : default_lb
    upper = hasmetadata(v, VariableUpperBound) ? getmetadata(v, VariableUpperBound) : default_ub
    push!(cell_lb, lower)
    push!(cell_ub, upper)
  end
  return cell_lb, cell_ub
end


include("layers.jl")
include("systems/ltc.jl")
include("systems/ltc_orig.jl")
include("systems/ltc_mtk.jl")
include("systems/ltc_gj.jl")
include("systems/ltc_syn_state.jl")
include("systems/ltc_syn_state_mtk.jl")
include("systems/stateful_mtk.jl")
include("systems/dLeak.jl")
include("wiring/basic.jl")
include("model.jl")
include("optimize.jl")
include("losses.jl")
include("callback.jl")
include("utils.jl")


export DataLoader

end
