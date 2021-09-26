mutable struct MyCallback{L,F,F2}
  losses::L
  cb::F
  ecb::F2
  nepochs::Int
  nsamples::Int
  iter::Int
  epoch::Int
  print_iter_loss::Bool
  print_epoch_mean_loss::Bool
  plot_iter::Bool
  plot_epoch::Bool
  MyCallback(losses, cb, ecb, nepochs, nsamples, iter, epoch, print_iter_loss, print_epoch_mean_loss, plot_iter, plot_epoch) =
    new{typeof(losses), typeof(cb), typeof(ecb)}(losses, cb, ecb, nepochs, nsamples, iter, epoch, print_iter_loss, print_epoch_mean_loss, plot_iter, plot_epoch)
end

MyCallback(T::DataType=Float32;
              cb=(args...;kwargs...)->false, ecb=(args...;kwargs...)->false,
              nepochs=1, nsamples=1, 
              print_iter_loss=false, plot_every_iter=false,
              print_epoch_mean_loss=true, plot_every_epoch=true) =
  MyCallback(T[], cb, ecb, nepochs, nsamples, 0, 0, print_iter_loss, print_epoch_mean_loss, plot_every_iter, plot_every_epoch)

function (mcb::MyCallback)(p,l, args...; kwargs...)
  cbout = invoke_sample_cb!(mcb, p,l, args...; kwargs...)
  mcb.iter % mcb.nsamples == 0 && invoke_epoch_cb!(mcb, p, l, args...; kwargs...)
  return cbout
end

function reset!(mcb::MyCallback)
  mcb.losses = Vector{eltype(mcb.losses)}(undef, 0)
  mcb.iter = 0
  nothing
end

function invoke_sample_cb!(mcb::MyCallback, p,l,args...; kwargs...)
  mcb.print_iter_loss == true && println(l)
  mcb.plot_iter == true && mcb.cb(p,l,args...; kwargs...)
  push!(mcb.losses, l)
  mcb.iter += 1
  return false
end

function invoke_epoch_cb!(mcb::MyCallback, p, l, args...; kwargs...)
  mcb.epoch += 1
  mcb.print_epoch_mean_loss == true && println("Epoch $(mcb.epoch) mean loss: $(mean(mcb.losses))")
  mcb.plot_epoch == true && mcb.ecb(p,l,args...; kwargs...)
  reset!(mcb)
  return false
end
