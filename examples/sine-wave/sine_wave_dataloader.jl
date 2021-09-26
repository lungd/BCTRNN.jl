import Flux: Data.DataLoader, stack

generate_2d_data(T) = generate_data(T; stacked=false)
generate_3d_data(T) = generate_data(T; stacked=true)
generate_3d_ct_data(T) = generate_data(T; stacked=true, ct=true)


function generate_ts_data(T::DataType=Float32)
  N = 48
  tsx = T.(collect(0:N-1))
  tsy = T.(collect(1:N))
  data_x = [sin.(range(0,stop=3π,length=N)), cos.(range(0,stop=3π,length=N))]
  data_x = [reshape([T(data_x[1][i]),T(data_x[2][i])],2,1) for i in 1:N]# |> f32
  data_y = [reshape([T(y)],1,1) for y in sin.(range(0,stop=6π,length=N))]# |> f32
  dl = DataLoader((zip(data_x,tsx), (data_y,tsy)), batchsize=N)
end

function generate_data(T::DataType=Float32; stacked=true, ct=false)
    in_features = 2
    out_features = 1
    N = 48
    data_x = [sin.(range(0,stop=3π,length=N)), cos.(range(0,stop=3π,length=N))]
    data_x = [reshape([T(data_x[1][i]),T(data_x[2][i])],2,1) for i in 1:N]# |> f32
    data_y = [reshape([T(y)],1,1) for y in sin.(range(0,stop=6π,length=N))]# |> f32
    dl = DataLoader((data_x, data_y), batchsize=N)
    @show length(dl)
    fx, fy = first(dl)
    @show size(fx)
    @show size(fx[1])
    @show size(fy)
    @show size(fy[1])
    fig = plot([x[1,1] for x in fx], label="x1")
    plot!(fig, [x[2,1] for x in fx], label="x2")
    plot!(fig, [y[1,1] for y in fy], label="y1")
    display(fig)

    if stacked == false && ct == false
      return dl
    end

    
    data_x, data_y = stacked == true ? (stack(data_x,2), stack(data_y,2)) : (data_x, data_y)
    saveat = collect(T(0):size(data_x,2)-1)
    #data_x, data_y = ct == true ? ((data_x, saveat), data_y) : (data_x, data_y)
    batchsize = stacked == true ? 1 : N
    batchsize = 48
    dx = [(data_x[:,s:s+batchsize-1,:],saveat[s:s+batchsize-1]) for s in 1:batchsize:size(data_x,2)-batchsize+1]
    dy = [data_y[:,s:s+batchsize-1,:] for s in 1:batchsize:size(data_y,2)-batchsize+1]

    @show summary(dx)
    @show summary(dy)
    dl = Flux.Data.DataLoader((dx, dy), batchsize=1)

    fxt, fy = first(dl)
    fxt, fy = fxt[1], fy[1]
    @show summary(fxt)
    @show summary(fy)
    fx,tx = fxt
    @show size(fx)
    @show size(tx)
    @show size(fy)
    @show size(fy)

    dl
end



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

