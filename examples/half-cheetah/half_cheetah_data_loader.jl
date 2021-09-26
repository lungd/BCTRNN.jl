import NPZ: npzread
using Flux
import Flux: stack, unstack, Data.DataLoader


get_2d_dl(T::DataType=Float32; seq_len=32, batchsize=16, normalise=true) = get_dl(T; seq_len, batchsize, normalise, stacked=false)
get_3d_dl(T::DataType=Float32; seq_len=32, batchsize=16, normalise=true) = get_dl(T; seq_len, batchsize, normalise, stacked=true)
get_3d_ct_dl(T::DataType=Float32; seq_len=32, batchsize=16, normalise=true, ct=true) = get_dl(T; seq_len, batchsize, normalise, stacked=true, ct=true)

function get_dl(T::DataType=Float32; seq_len=32, batchsize=16, stacked=true, normalise=true, ct=false)
    filepath = joinpath(@__DIR__, "data")
    data_dir = filepath

    obs_size = size(npzread("$(data_dir)/trace_0000.npy"),2) # 17

    all_files = readdir(data_dir)
    all_files = ["$(data_dir)/$(f)" for f in all_files]
    valid_files = all_files[1:5]
    test_files = all_files[6:15]
    train_files = all_files[16:16]




    train_x, train_y, train_seq = _load_files(T, train_files, seq_len)
    test_x, test_y, test_seq = _load_files(T, test_files, seq_len)
    valid_x, valid_y, valid_seq = _load_files(T, valid_files, seq_len)

    # (970×32×17)
    train_x = permutedims(train_x,(3,1,2))
    train_y = permutedims(train_y,(3,1,2))
    test_x = permutedims(test_x,(3,1,2))
    test_y = permutedims(test_y,(3,1,2))
    valid_x = permutedims(valid_x,(3,1,2))
    valid_y = permutedims(valid_y,(3,1,2))


    # (17x970x32)
    if normalise
      train_x = Flux.normalise(train_x)
      train_y = Flux.normalise(train_y)
    end

    train_x_new = train_y_new = stacked ? Vector{Array{T,3}}() : Vector{Vector{Matrix{T}}}()
    test_x_new = test_y_new = stacked ? Vector{Array{T,3}}() : Vector{Vector{Matrix{T}}}()
    valid_x_new = valid_y_new = stacked ? Vector{Array{T,3}}() : Vector{Vector{Matrix{T}}}()

    if ct
      train_x_new = train_y_new = Vector{Tuple{Array{T,3},Tuple{T,T}}}()
      test_x_new = test_y_new = Vector{Tuple{Array{T,3},Tuple{T,T}}}()
      valid_x_new = valid_y_new = Vector{Tuple{Array{T,3},Tuple{T,T}}}()
    end
    # train_x_new = train_y_new = []
    # test_x_new = test_y_new = []
    # valid_x_new = valid_y_new = []
    for i in 1:batchsize:size(train_x,2)-batchsize-1
        tmp_x = unstack(train_x[:,i:i+batchsize-1,:],3)
        tmp_y = unstack(train_y[:,i+1:i+batchsize,:],3)
        tmp_x, tmp_y = stacked ? (stack(tmp_x,2),stack(tmp_y,2)) : (tmp_x,tmp_y)

        if ct
          tspan_x = (T(i-1),T(i-1+size(tmp_x,2)))
          tspan_y = (T(i),T(i+size(tmp_y,2)))
          tmp_x, tmp_y = (tmp_x, tspan_x), (tmp_y, tspan_y)
        end
        push!(train_x_new, tmp_x)
        push!(train_y_new, tmp_y)
        # push!(train_x_new, train_x[:,i:i+batchsize-1,:])
        # push!(train_y_new, train_y[:,i+1:i+batchsize,:])
    end
    for i in 1:batchsize:size(test_x,2)-batchsize-1
        tmp_x = unstack(test_x[:,i:i+batchsize-1,:],3)
        tmp_y = unstack(test_y[:,i+1:i+batchsize,:],3)
        tmp_x, tmp_y = stacked ? (stack(tmp_x,2),stack(tmp_y,2)) : (tmp_x,tmp_y)
        if ct
          tspan_x = (T(i-1),T(i-1+size(tmp_x,2)))
          tspan_y = (T(i),T(i+size(tmp_y,2)))
          tmp_x, tmp_y = (tmp_x, tspan_x), (tmp_y, tspan_y)
        end
        push!(test_x_new, tmp_x)
        push!(test_y_new, tmp_y)
    end
    for i in 1:batchsize:size(valid_x,2)-batchsize-1
        tmp_x = unstack(valid_x[:,i:i+batchsize-1,:],3)
        tmp_y = unstack(valid_y[:,i+1:i+batchsize,:],3)
        tmp_x, tmp_y = stacked ? (stack(tmp_x,2),stack(tmp_y,2)) : (tmp_x,tmp_y)
        if ct
          tspan_x = (T(i-1),T(i-1+size(tmp_x,2)))
          tspan_y = (T(i),T(i+size(tmp_y,2)))
          tmp_x, tmp_y = (tmp_x, tspan_x), (tmp_y, tspan_y)
        end
        push!(valid_x_new, tmp_x)
        push!(valid_y_new, tmp_y)
    end

    # train_x_new = Flux.stack(train_x_new,4)
    # train_y_new = Flux.stack(train_y_new,4)
    # test_x_new = Flux.stack(test_x_new,4)
    # test_y_new = Flux.stack(test_y_new,4)
    # valid_x_new = Flux.stack(valid_x_new,4)
    # valid_y_new = Flux.stack(valid_y_new,4)

    #train_x = Flux.unstack(train_x,3)

    # train_dl = Flux.Data.DataLoader((train_x,train_y),batchsize=1)
    # test_dl = Flux.Data.DataLoader((test_x,test_y),batchsize=batchsize)
    # valid_dl = Flux.Data.DataLoader((valid_x,valid_y),batchsize=batchsize)
    # train_dl = Flux.Data.DataLoader((train_x_new,train_y_new),batchsize=1)
    # test_dl = Flux.Data.DataLoader((test_x_new,test_y_new),batchsize=1)
    # valid_dl = Flux.Data.DataLoader((valid_x_new,valid_y_new),batchsize=1)

    # @show size(train_x_new)
    # @show size(train_x_new[1])
    # @show size(train_x_new[1][1])
    #
    # @show size(train_x_new[2])
    # @show size(train_x_new[2][1])

    train_dl = zip(train_x_new, train_y_new) #|> f32
    test_dl = zip(test_x_new, test_y_new) #|> f32
    valid_dl = zip(valid_x_new, valid_y_new) #|> f32

    return train_dl, test_dl, valid_dl, train_x_new

    @show summary(train_x_new)
    @show summary(train_x_new[1])
    @show summary(train_x_new[1][1])

    #train_dl = DataLoader((train_x_new, train_y_new), batchsize=1, shuffle=false)
    # i = 0
    # for (tx,ty) in train_dl
    #   @show tx[1][2]
    #   i += 1
    #   if i == 3
    #     break
    #   end
    # end


    # train_dl = (train_x_new, test_y_new)
    # test_dl = (test_x_new, test_y_new)
    # valid_dl = (valid_x_new, valid_y_new)

    return train_dl, test_dl, valid_dl, train_seq
end


function _load_files(T, files, seq_len)
    all_x = []
    all_y = []
    sequences = []
    for f in files
        arr = T.(npzread(f))
        push!(sequences, permutedims(arr, (2,1)))
        x, y = _cut_in_sequences(arr, seq_len, 10)
        push!(all_x, x...)
        push!(all_y, y...)
    end

    return stack(all_x,1), stack(all_y,1), sequences
    #return all_x, all_y
end

function _cut_in_sequences(x, seq_len, inc=1)
    seq_x = []
    seq_y = []
    for s in 1:inc:size(x,1)-seq_len-1
        i_s = s
        i_e = i_s + seq_len - 1
        push!(seq_x, x[i_s:i_e,:])
        push!(seq_y, x[i_s+1:i_e+1,:])
    end
    return seq_x, seq_y
end


# train_dl = get_dl()[1][1]
# x = train_dl[1]











function mycb(p,l,ŷ,y;doplot=true)
  #display(l)
  if doplot
    y = ndims(y) < 3 ? stack(y,2) : y
    ŷ = ndims(ŷ) < 3 ? stack(ŷ,2) : ŷ

    if size(y,1) > 3
      hy = heatmap(1:size(y,2),1:size(y,1),(@view y[:,:,1]),title="data")
      hŷ = heatmap(1:size(ŷ,2),1:size(ŷ,1),(@view ŷ[:,:,1]),title="prediction")
      he = heatmap(1:size(y,2),1:size(y,1),(@view y[:,:,1]).-(@view ŷ[:,:,1]),title="error", c=:curl)
      p = plot((@view y[1,:,1]), label="y[1,:,1]")
      plot!(p, (@view ŷ[1,:,1]), label="ŷ[1,:,1]")
      fig = plot(hy,he,hŷ,p)
      display(fig)
    else
      p = plot((@view y[1,:,1]), label="y1")
      plot!(p, (@view ŷ[1,:,1]), label="ŷ1")
      for i in 2:size(y,1)
        plot!(p, (@view y[i,:,1]), label="y$(i)")
        plot!(p, (@view ŷ[i,:,1]), label="ŷ$(i)")
      end
      display(p)

      if size(y,3) > 1
        p = plot((@view y[1,:,2]), label="y1")
        plot!(p, (@view ŷ[1,:,2]), label="ŷ1")
        for i in 2:size(y,1)
          plot!(p, (@view y[i,:,2]), label="y$(i)")
          plot!(p, (@view ŷ[i,:,2]), label="ŷ$(i)")
        end
        display(p)
      end
    end

    # fig = plot(y[1,:,1], label="y1")
    # size(y,1) > 1 && plot!(fig, y[2,:,1], label="y2")
    # plot!(fig, ŷ[1,:,1], label="ŷ1")
    # size(y,1) > 1 && plot!(fig, ŷ[2,:,1], label="ŷ2")
    # display(fig)
    #
    # fig = plot(y[end-1,:,1], label="ye1")
    # size(y,1) > 1 && plot!(fig, y[end,:,1], label="ye2")
    # plot!(fig, ŷ[end-1,:,1], label="ŷe1")
    # size(y,1) > 1 && plot!(fig, ŷ[end,:,1], label="ŷe2")
    # display(fig)
  end
  return false
end

