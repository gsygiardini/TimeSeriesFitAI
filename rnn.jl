using Flux
using Statistics
using CSV
using DataFrames
using Plots
using Zygote
# using PyPlot

function dx(t,M,λ,μ)
#     α  = 0.3
#     σ² = 0.4
#     λ  = α/t
#     μ  = t/σ²

    if isinteger(t)
        t = ceil(Int,t)
        return (λ[t] - μ[t])*M
    else
        t = ceil(Int,t)
        λ_int = (λ[t] + λ[t+1])/2
        μ_int = (μ[t] + μ[t+1])/2
        return (λ_int - μ_int)*M
    end
end

function RK(t,M,λ,μ,Δ)
    K1 = dx(t,M,λ,μ)
    K2 = dx(t + 0.5*Δ, M .+ 0.5*Δ*K1, λ, μ)
    K3 = dx(t + 0.5*Δ, M .+ 0.5*Δ*K2, λ, μ)
    K4 = dx(t + Δ, M .+ Δ*K3, λ, μ)

    return M + Δ*(K1 .+ 2*K2 .+ 2*K3 + K4)/6
end

function M_t(T,λ,μ)
    Δ = 0.01
    t = Δ
    M = 1E-6
    y = zeros(T)  # Preallocate an array with a fixed size

    for it in 1:T
        M = RK(t,M,λ,μ,Δ)
        t += Δ
        y[it] = M  # Assign the value to the specific index
    end
    return copy(y/maximum(y))
end

function genData(T)
    data = CSV.read("citations.csv", DataFrame)
    x_axis = Float32.(data.citation_1_x)
    y_axis = Float32.(data.citation_1_y)

    y_axis = y_axis[2:end]

    epochs = 10
    Δ = 0.01
    t = Δ
    M = 1E-6
    y = []
    α  = 0.3
    σ² = 0.4
    λ_pre  = [α/t for t ∈ 1:size(x_axis)[1]]
    μ_pre  = [t/σ² for t ∈ 1:size(x_axis)[1]]

    λ = [rand() for i ∈ 1:size(x_axis)[1]]
    μ = [rand() for i ∈ 1:size(x_axis)[1]]

    # Pre train functions
#     λ = train(λ, λ_pre, 100, :mse)
#     μ = train(μ, μ_pre, 100, :mse)

    #Testing the trained output
    print(train_RK(λ, y_axis, μ, 10, :r2))

#     scatter(μ, label="Predicted")
#     scatter!(μ_pre, label="Actual")
#     savefig("fig.png")

end

function train_RK(γ_train, γ_real, aux, epochs, optimize::Symbol)
#     γ_train = [[χ] for χ in χ_train] # Reshape input data into Flux recurrent data format
#     γ_real = χ_real

    model = Chain(
        RNN(1 => 32, relu),
        Dense(32 => 1, identity) # Create a recurrent model
    )

    opt = ADAM() # Select the optimizer function from Flux

    θ = Flux.params(model) # Keep track of the model parameters

#     model(γ_train[1]) # Warm-up the model

    epochs = 2
    for epoch in 1:epochs
        Flux.reset!(model) # Reset the hidden state of the RNN

        ∇ = gradient(θ) do
#             γ_pred = M_t(44, χ_train, aux) # Calculate γ_pred

            ssr = sum([(M_t(44, γ_train, aux)[i] - γ_real[i])^2 for i in 1:43])
            sst = sum([(γ_real[i] - mean(γ_real))^2 for i in 1:43])

            1.0 - ssr / sst # Calculate R2 score
        end

        Flux.update!(opt, θ, ∇) # Update the parameters
    end
#     result = [model(γ)[1] for γ ∈ γ_train[2:end]]
#     return vcat(result...)
end

##################################################################################
################################## Main Program ##################################
##################################################################################
genData(10)
