using Distributed, Random, Clustering, StatsPlots, DataFrames, LinearAlgebra
@everywhere using CSV, SharedArrays

#The same as the Python one, uses arbitrary precision
@everywhere function dist_quakes(sismo_i, sismo_j, c, B)
 tau = abs(sismo_j[5] - sismo_i[5])
 lat1 = sismo_i[2]
 lat2 = sismo_j[2]
 lon1 =  sismo_i[3]
 lon2 = sismo_j[3]

 rad = pi/180
 dlat = lat2-lat1
 dlon = lon2-lon1
 R = 6372.795477598
 a = (sin(rad*dlat/2))^2 + cos(rad*lat1)*cos(rad*lat2)*(sin(rad*dlon/2))^2
 distancia = 2*R*asin(sqrt(a))

 r = sqrt(distancia^2 + (sismo_j[4]-sismo_i[4])^2 )
 delta_m = convert(BigFloat, sismo_j[1] - sismo_i[1])
 return c*tau*(r^(2*B) )* (10^(-B * delta_m))
end


train = CSV.read("/Users/bernardoflores/Documents/ITT-CENAPRED-6/code/train.csv", DataFrame)
#To do parallel processing
train = SharedArray(Matrix(train]))

#Only the upper triangular part will be filled
distances = zeros(BigFloat, 6000, 6000)
@sync @distributed for i = 1:6000
  for j = (i+1):6000
    distances[i,j] = dist_quakes(train[i,:], train[j,:], 1, 0.6160988624731996)
  end
end

distances = LinearAlgebra.Symmetric(distances, :U)
distances = Matrix{Float64}(distances)
CSV.write("/Users/bernardoflores/Documents/ITT-CENAPRED-6/code/distances.csv", DataFrame(distances, :auto))

### Clustering

Random.seed!(3)

tree = hclust(distances, linkage=:complete)
plot(tree, branchorder=:optimal)
savefig("/Users/bernardoflores/Documents/ITT-CENAPRED-6/code/dendogram_6000.pdf")
#Too many leaves

tree_1000 = hclust(distances[1:1000, 1:1000], linkage=:complete)
plot(tree_1000, branchorder=:optimal)
savefig("/Users/bernardoflores/Documents/ITT-CENAPRED-6/code/dendogram_1000.pdf")
#Still too many

tree_100 = hclust(distances[1:100, 1:100], linkage=:complete)
plot(tree_100, branchorder=:optimal)
savefig("/Users/bernardoflores/Documents/ITT-CENAPRED-6/code/dendogram_100.pdf")

# perhaps 3 clusters?
