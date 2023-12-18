using Serialization
using Statistics
using DataFrames
using CSV

# Anggota:
# Kenneth Sabastian (1313621004)
# Muhammad Faris Heruputra (1313621014)


function find_mean(data_matrix::Vector{Float16},class_matrix::Vector{Float16},class_type::Vector{Float16})::Vector{Float16}
    means = zeros(Float16,3)
    @simd for i in eachindex(class_type)
        class_data = convert(Vector{Float32},view(data_matrix,findall(==(class_type[i]),class_matrix)))
        @fastmath means[i] = mean(class_data)
        class_data = 0
    end
    class_type = 0
    return means
end

function find_mean(data_matrix::Matrix{Float16},class_matrix::Vector{Float16},class_type::Vector{Float16})::Matrix{Float16}
    means = zeros(Float16,3,4)
    @simd for i in eachindex(class_type)
        class_data = convert(Matrix{Float32},view(data_matrix,findall(==(class_type[i]),class_matrix),:))
        @fastmath means[i,:] = mean(class_data,dims = 1)
        class_data = 0
    end
    class_type = 0
    return means
end

function prediction(data_matrix::Vector{Float16},class_matrix_real::Vector{Float16},means::Vector{Float16})
    class_matrix = copy(class_matrix_real)
    @simd for i in eachindex(class_matrix)
        @fastmath class_matrix[i] += distance(means,data_matrix[i]) * 10
    end
    result = zeros(Int,3,3)
    @simd for i in 1:3
        @simd for j in 1:3
            result[i,j] = count(==(i*10+j),class_matrix)
        end
    end
    class_matrix = 0
    return result
end

function prediction(data_matrix::Matrix{Float16},class_matrix_real::Vector{Float16},means::Matrix{Float16})
    class_matrix = copy(class_matrix_real)
    @simd for i in eachindex(class_matrix)
        @fastmath @inbounds class_matrix[i] += distance(means,data_matrix[i,:]) * 10
    end
    result = zeros(Int,3,3)
    @simd for i in 1:3
        @simd for j in 1:3
            @inbounds result[i,j] = count(==(i*10+j),class_matrix)
        end
    end
    class_matrix = 0
    return result
end

function distance(means::Vector{Float16},data::Float16)
    result = zeros(Float16,3)
    @simd for i = 1:3
        result[i] = (data - means[i])^2
    end
    return argmin(result)
end

function distance(means::Matrix{Float16},data::Vector{Float16})
    result = zeros(Float16,3)
    @simd for i = 1:3
        @simd for j = 1:4
            result[i] += (data[j] - means[i,j])^2
        end
    end
    return argmin(result)
end

function apply_cascade(data_matrix::Matrix{Float16},class_matrix::Vector{Float16},means::Matrix{Float16},stdev::Matrix{Float16},class::Int64)
    #println(size(data_matrix)[1]," " ,size(class_matrix)[1])
    copy_class_matrix = copy(class_matrix)
    @simd for i in eachindex(class_matrix)
        @fastmath copy_class_matrix[i] += distance(means[:,class],data_matrix[i,class]) * 10
    end
    @simd for i in 1:3
        index =  findall(==(11*i),copy_class_matrix)
        process_matrix = convert(Vector{Float32},data_matrix[index,class])
        means[i,class] = mean(process_matrix)
        stdev[i,class] = std(process_matrix)
        data_matrix = data_matrix[setdiff(1:end,index),:]
        copy_class_matrix = copy_class_matrix[setdiff(1:end,index)]
        class_matrix = class_matrix[setdiff(1:end,index)]
        index = []
        #println(size(data_matrix)[1]," ",size(class_matrix)[1])
    end
    copy_class_matrix = 0
    return [data_matrix,class_matrix,means,stdev]
end

function guess_distance(means::Vector{Float16},data::Float16,stdev::Vector{Float16})
    result = zeros(Float16,3)
    @simd for i = 1:3
        result[i] = (data - means[i])^2
    end
    index = sortperm(result)
    #if result[index[1]] < result[index[2]]*0.2
    #if result[index[1]] < stdev[index[1]] && result[index[2]] > stdev[index[2]]
    if result[index[1]] < stdev[index[1]]^2
    #if result[index[2]] - result[index[1]] > stdev[index[1]]
        return index[1]
    else
        return 0
    end
end

function guess(data_matrix::Matrix{Float16},class_matrix::Vector{Float16},means::Matrix{Float16},stdev::Matrix{Float16},order::Vector{Int64})
    result = zeros(Int,3,3)
    for i in order
        @simd for j in eachindex(class_matrix)
            prediction = guess_distance(means[:,i],data_matrix[j,i],stdev[:,i])
            if prediction !=0
                @fastmath class_matrix[j] += prediction * 10
            end
        end 
        @simd for k in 1:3
            @simd for l in 1:3
                index =  findall(==(k*10+l),class_matrix)
                result[k,l] += size(index)[1]
                data_matrix = data_matrix[setdiff(1:end,index),:]
                class_matrix = class_matrix[setdiff(1:end,index)]
            end
        end
    end
    for i in eachindex(class_matrix)
        @fastmath class_matrix[i] += distance(means,data_matrix[i,:]) * 10
    end
    @simd for i in 1:3
        @simd for j in 1:3
            result[i,j] += count(==(i*10+j),class_matrix)
        end
    end
    return result
end

file = deserialize("$(@__DIR__)/data_9m.mat")
#file = Matrix{Float16}(DataFrame(CSV.File(open("$(@__DIR__)/iris.csv"))))
#display(file[end-100:end-1,:])
data_matrix = file[:,1:4]
row = size(data_matrix,1)
class_matrix = file[:,5]
class_type = unique(class_matrix)
file = 0
order_value = []

for i in 1:4
    data = data_matrix[:,i]
    means_local = find_mean(data,class_matrix,class_type)
    result_local = prediction(data,class_matrix,means_local)
    result2_local = zeros(Float64,3)
    @simd for i in 1:3
        result2_local[i] = result_local[i,i]/sum(result_local[:,i])
    end
    println("Data dengan kolom $(i): ")
    println(result2_local)
    println("persentase: $(sum(result2_local))")
    display(means_local)
    println("")
    percentage = sum(result2_local)
    push!(order_value,percentage)
end
means = find_mean(data_matrix,class_matrix,class_type)
result_euclidian = prediction(data_matrix,class_matrix,means)
result2_euclidian = zeros(Float64,3)
@simd for i in 1:3
    result2_euclidian[i] = result_euclidian[i,i]/sum(result_euclidian[:,i])
end
println("Data dengan euclidian distance semua kolom: ")
display(result_euclidian)
println(result2_euclidian)
println(sum(result2_euclidian))
println("")
order = sortperm(order_value, rev = true)
stdev = zeros(Float16,3,4)
copy_data_matrix = copy(data_matrix)
copy_class_matrix = copy(class_matrix)
println("Cascade order:")
println(order)
println("")
println("Means before cascade:")
display(means)
println("")
for i in eachindex(order)
    result = apply_cascade(copy_data_matrix,copy_class_matrix,means,stdev,order[i])
    global copy_data_matrix = result[1]
    global copy_class_matrix = result[2]
    global means = result[3]
    global stdev = result[4]
end
println("Means and stdev after cascade:")
display(means)
display(stdev)
println("")
#result = prediction(data_matrix,class_matrix,means)
println("Data dengan cascading semua kolom: ")
result = guess(data_matrix,class_matrix,means,stdev,order)
display(result)
result2 = zeros(Float64,3)
@simd for i in 1:3
    result2[i] = result[i,i]/sum(result[:,i])
end
println(result2)
println(sum(result2))