function _get_bounds(T, default_lb, default_ub, vars)
  cell_lb = T[]
  cell_ub = T[]
  for v in vars
    contains(string(v), "InPin") && continue
    contains(string(v), "OutPin") && continue
    hasmetadata(v, ModelingToolkit.VariableOutput) && continue
    lower = hasmetadata(v, VariableLowerBound) ? getmetadata(v, VariableLowerBound) : default_lb
    upper = hasmetadata(v, VariableUpperBound) ? getmetadata(v, VariableUpperBound) : default_ub
    push!(cell_lb, lower)
    push!(cell_ub, upper)
  end
  return cell_lb, cell_ub
end


function print_cell_info(cell, wiring, train_u0)
  println("--------------- MTK Cell ---------------")
  println("in:                         $(wiring.n_in)")
  println("out:                        $(wiring.n_out)")
  println("# neurons:                  $(wiring.n_total)")
  println("# input-neuron synapses:    $(Int(sum(wiring.sens_mask)))")
  println("# neuron-neuron synapses:   $(Int(sum(wiring.syn_mask)))")
  println("# params:                   $(length(cell.p))")
end
