function get_model(re::Chain, p)
  m = re(p)
  reset_state!(m, p)
  (x, p) -> m(x)
end
function get_model(m::FastChain, p)
  reset_state!(m, p)
  m
end

function get_model(re, p)
  m = re(p)
  reset_state!(m, p)
  (x, p) -> m(x)
end

function loss_seq(p, m, x, y)
  m = get_model(m,p)

  ŷb = Flux.Zygote.Buffer([y[1]], size(y,1))
  @inbounds for i in 1:size(x,1)
    xi = x[i]
    ŷi = m(xi, p)
    (Inf32 ∈ ŷi || NaN ∈ ŷi) && return Inf32, y, y # TODO: what if a layer after MTKRecur can handle Infs?
    ŷb[i] = ŷi
  end
  ŷ = copy(ŷb)

  # ŷ = m.(x,[p])
  # Inf32 ∈ ŷ && return Inf32, ŷ, y

  return mean(Flux.Losses.mse.(ŷ,y, agg=mean)), ŷ, y
end

# function loss_3d_seq(p, m, x, y)
#   m = get_model(m,p)
#   ŷ = m(x, p)
#   Inf ∈ ŷ && return Inf32, ŷ, y
#   l = mean(Flux.Losses.mse.(ŷ,y, agg=mean))
#   return l, ŷ, y
# end