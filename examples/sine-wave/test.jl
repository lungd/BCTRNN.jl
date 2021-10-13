using Pkg, DiffEqSensitivity, OrdinaryDiffEq, Zygote, ComponentArrays, ReverseDiff
using Parameters
Pkg.status("DiffEqSensitivity")
# DiffEqSensitivity v6.58.0

## Proof that ReverseDiff works with ComponentArrays
ReverseDiff.gradient(x->sum(x),ComponentArray(a=1.,b=7.,c=8.))
# ComponentVector{Float64}(a = 1.0, b = 1.0, c = 1.0)


## Proof that ReverseDiff for ODEs doesn't work with ComponentArrays
u0 = [1.0;1.0]
p = ComponentArray(a=rand(3),b=1.0,_c=rand(2))
function fiip(du,u,p,t)
  @unpack a, b, _c = p
  c,d = _c
  du[1] = dx = sum(a)*u[1] - b*u[1]*u[2]
  du[2] = dy = -c*u[2] + d*u[1]*u[2]
end

function foop(u,p,t)
  @unpack a, b, _c = p
  c,d = _c
  dx = sum(a)*u[1] - b*u[1]*u[2]
  dy = -c*u[2] + d*u[1]*u[2]
  [dx, dy]
end
prob = ODEProblem(foop,u0,(0.0,10.0),p)

prob2 = ODEProblem(fiip,u0,(0.0,10.0),p)

# ForwardDiff works
loss(p) = sum(solve(prob,Tsit5(),u0=u0,p=p,saveat=0.1,sensealg=ForwardDiffSensitivity()))
dp = Zygote.gradient(loss,p)[1]
# ComponentVector{Float64}(a = 7.349039781610272, b = -159.31079871982794, c = 74.93924771425637, d = -339.3272371527868)

# ReverseDiff fails
loss2(p) = sum(solve(prob,Tsit5(),u0=u0,p=p,saveat=0.1,sensealg=ReverseDiffAdjoint()))
dp2 = Zygote.gradient(loss2,p)[1]


### OOP
# some zeros
loss3(p) = sum(solve(prob,Tsit5(),u0=u0,p=p,saveat=0.1,sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))
dp3 = Zygote.gradient(loss3,p)[1]

#  24.963 ms (200048 allocations: 10.48 MiB)
loss33(p) = sum(solve(prob,Tsit5(),u0=u0,p=p,saveat=0.1,sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP())))
@btime dp33 = Zygote.gradient(loss33,p)[1]

#  25.901 ms (199973 allocations: 10.48 MiB)
loss4(p) = sum(solve(prob,Tsit5(),u0=u0,p=p,saveat=0.1,sensealg=InterpolatingAdjoint()))
@btime dp4 = Zygote.gradient(loss4,p)[1]

#  25.885 ms (204438 allocations: 10.60 MiB)
loss5(p) = sum(solve(prob,Tsit5(),u0=u0,p=p,saveat=0.1,sensealg=BacksolveAdjoint()))
@btime dp5 = Zygote.gradient(loss5,p)[1]

#  29.091 ms (217545 allocations: 11.31 MiB)
loss55(p) = sum(solve(prob,Tsit5(),u0=u0,p=p,saveat=0.1,sensealg=BacksolveAdjoint(autojacvec=ZygoteVJP())))
@btime dp55 = Zygote.gradient(loss55,p)[1]

#  69.483 ms (565278 allocations: 29.98 MiB)
loss6(p) = sum(solve(prob,Tsit5(),u0=u0,p=p,saveat=0.1,sensealg=QuadratureAdjoint()))
@btime dp6 = Zygote.gradient(loss6,p)[1]

#  73.882 ms (573837 allocations: 30.47 MiB)
loss7(p) = sum(solve(prob,Tsit5(),u0=u0,p=p,saveat=0.1,sensealg=QuadratureAdjoint(autojacvec=ZygoteVJP())))
@btime dp7 = Zygote.gradient(loss7,p)[1]





### IP

# some zeros
loss3(p) = sum(solve(prob2,Tsit5(),u0=u0,p=p,saveat=0.1,sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP(true))))
dp3 = Zygote.gradient(loss3,p)[1]

#  zeros
loss33(p) = sum(solve(prob2,Tsit5(),u0=u0,p=p,saveat=0.1,sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP())))
@btime dp33 = Zygote.gradient(loss33,p)[1]

#  some zeros
loss4(p) = sum(solve(prob2,Tsit5(),u0=u0,p=p,saveat=0.1,sensealg=InterpolatingAdjoint()))
@btime dp4 = Zygote.gradient(loss4,p)[1]

#  some zeros
loss5(p) = sum(solve(prob2,Tsit5(),u0=u0,p=p,saveat=0.1,sensealg=BacksolveAdjoint()))
@btime dp5 = Zygote.gradient(loss5,p)[1]

#  
loss6(p) = sum(solve(prob2,Tsit5(),u0=u0,p=p,saveat=0.1,sensealg=QuadratureAdjoint()))
@btime dp6 = Zygote.gradient(loss6,p)[1]

loss7(p) = sum(solve(prob2,Tsit5(),u0=u0,p=p,saveat=0.1,sensealg=QuadratureAdjoint(autojacvec=ZygoteVJP())))
@btime dp7 = Zygote.gradient(loss7,p)[1]