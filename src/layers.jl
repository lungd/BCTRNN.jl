rand_uniform(T::DataType, lb,ub,dims...) = T.(rand(Uniform(T(lb),T(ub)),dims...))
rand_uniform(T::DataType, lb,ub) = rand_uniform(T, lb,ub,1)[1]
rand_uniform(lb,ub,dims...) = rand_uniform(Float32, lb,ub,dims...)
rand_uniform(lb,ub) = rand_uniform(Float32, lb,ub, 1)[1]
Zygote.@nograd rand_uniform
add_dim(x::Array{T, N}) where {T,N} = reshape(x, Val(N+1))

mysigm(x) = one(x) / (one(x) + exp(-x))
#mysigm(x) = NNlib.σ(x)
#ModelingToolkit.@register NNlib.σ


struct Mapper{W,B,F,P}
  W::W
  b::B
  σ::F
  p::P
  function Mapper(W, b, σ, p)
    new{typeof(W),typeof(b),typeof(σ),typeof(p)}(W, b, σ, p)
  end
end
function Mapper(n::Integer,σ::Function=identity; T=Float32, init=dims->ones(T,n), bias=dims->zeros(T,n))
  W = init(n)
  b = bias(n)
  p = vcat(W,b)
  Mapper(W, b, σ, p)
end
function (m::Mapper)(x::AbstractMatrix, p=m.p)
  Wl = length(m.W)
  W = @view p[1 : Wl]
  b = @view p[Wl + 1 : end]
  m.σ.(W .* x .+ b)
end
Base.show(io::IO, m::Mapper) = print(io, "Mapper(", m.W, ", ", m.σ, ")")
initial_params(m::Mapper) = m.p
paramlength(m::Mapper) = length(m.p)
Flux.functor(::Type{<:Mapper}, m) = (m.p,), re -> Mapper(m.W, m.b, m.σ, re...)
Flux.trainable(m::Mapper) = (m.p,)
function get_bounds(m::Mapper, T::DataType=eltype(m.p))
  lb = T.(vcat(fill(-10.1, length(m.W)), fill(-10.1, length(m.b))))
  ub = T.(vcat(fill(10.1, length(m.W)), fill(10.1, length(m.b))))
  lb, ub
end



paramlength(m::Flux.Dense) = length(m.weight) + (m.bias != Flux.Zeros() ? length(m.bias) : 0)
paramlength(m::Union{Flux.Chain,FastChain}) = sum([paramlength(l) for l in m.layers])
#paramlength(m::AbstractFluxLayer) = length(Flux.destructure(m)[1])

function get_bounds(l, T::DataType)
  p, _ = Flux.destructure(l)
  fill(T(-Inf), length(p)), fill(T(Inf), length(p))
end
function get_bounds(m::Union{Flux.Chain, FastChain}, T::DataType)
  lb = vcat([get_bounds(layer, T)[1] for layer in m.layers]...)
  ub = vcat([get_bounds(layer, T)[2] for layer in m.layers]...)
  lb, ub
end
function get_bounds(m::FastDense, T::DataType=nothing)
  lb = T.(vcat(fill(-10.1, m.out*m.in), fill(-10.1, m.out)))
  ub = T.(vcat(fill(10.1, m.out*m.in), fill(10.1, m.out)))
  if !m.bias
    lb = T.(vcat(fill(-10.1, m.out*m.in)))
    ub = T.(vcat(fill(10.1, m.out*m.in)))
  end
  lb, ub
end

function get_bounds(m::Flux.Dense{F, <:AbstractMatrix{T}, B}, ::DataType=nothing) where {F,T,B}
  lb = T.(vcat(fill(-10.1, length(m.weight)), fill(-10.1, length(m.bias))))
  ub = T.(vcat(fill(10.1, length(m.weight)), fill(10.1, length(m.bias))))
  if m.bias == Flux.Zeros()
    lb = T.(vcat(fill(-10.1, length(m.weight))))
    ub = T.(vcat(fill(10.1, length(m.weight))))
  end
  lb, ub
end


function reset_state!(m::Union{Flux.Chain, FastChain}, p)
  start_idx = 1
  for l in m.layers
    pl = paramlength(l)
    p_layer = @view p[start_idx:start_idx+pl-1]
    reset_state!(l, p_layer)
    start_idx += pl
  end
end

reset_state!(m,p) = nothing
