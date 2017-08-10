#!/usr/bin/python3.4
# DBSCAN implementation
# (c) Mohammad HMofrad, 2017 
# (e) mohammad.hmofrad@pitt.edu



import numpy as np
from utils import *


def region_query(points, point_index, epsilon):
    current_point = np.tile(points[point_index,:],(n,1))
    distance = np.linalg.norm(current_point - points, axis=1)
    p = np.arange(n)
    neighbor_points = p[(distance <= epsilon)]
    return neighbor_points

def expand_cluster(points, point_index, neighbor_points, clusters, cid, epsilon, min_points):	
    clusters[point_index] = cid
    k = 0
    while(True):
        point = neighbor_points[k]
        if(visited[point] == 0):
            visited[point] = 1
            neighbor_points_ = region_query(x, point, epsilon)	
            if(neighbor_points_.size >= min_points):
                neighbor_points = np.append(neighbor_points, np.setdiff1d(neighbor_points_,neighbor_points))		
        k = k + 1
        if(k == neighbor_points.size):
            break
        if(clusters[point] <= 0):
            clusters[point] = cid

np.random.seed()
# Read and store the input data
# using the utils.py
PERFIX = 'dataset/'
#FILE = PERFIX + 'balance-scale.data.txt'
#FILE = PERFIX + 'breast-cancer-wisconsin.data.txt'
#FILE = PERFIX + 'sonar.all-data.txt'
#FILE = PERFIX + 'cmc.data.txt'
#FILE = PERFIX + 'glass.data.txt'
#FILE = PERFIX + 'hayes-roth.data.txt'
#FILE = PERFIX + 'ionosphere.data.txt'
FILE = PERFIX + 'iris.data.txt'
#FILE = PERFIX + 'pima-indians-diabetes.data.txt'
#FILE = PERFIX + 'wine.data.txt'
#FILE = PERFIX + 'drift.data.txt'
#FILE = PERFIX + 'har.data.txt'

[x, y] = read(FILE)

# Initliaze parameters
[n, d] = np.shape(x)   # [#samples, #dimensions]
clusters = -np.ones(n)        # Cluster membership 
cid = 0               # Cluster id
#k = len(np.unique(y))  #  #clusters
visited = np.zeros(n)  # Visited

epsilon = .4

min_points = d - 1
starting_point = np.random.randint(0,n, 1)[0]
#sentinel = j

#while(True):
#for j in range(n):
#jj = (j % n)	
    
    
#print(j, jj)	
    #print(j, jj, (n % j), (j % n), ((jjj - j) % n))
#j = j + 1
    #jj = jjj + (n % j)
    #jjj = jjj + (j % n)
	#(jjj - j) % n
    #jjjj = jjjj + 1
#if((j % n) == sentinel):
#    break

#print('++++++++++++++++++==')
for i in range(starting_point, starting_point + n):
    j = i - starting_point
#    print(j)
    if(visited[j] == 1):
        pass
    else:
        visited[j] = 1
        neighbor_points = region_query(x, j, epsilon)
        if(neighbor_points.size < min_points):
            clusters[j] = -cid
        else: # Expand the cluster
            cid = cid + 1
            expand_cluster(x, j, neighbor_points, clusters, cid, epsilon, min_points)
    
#exit(0)
print(clusters)

for i in range(n):
    if clusters[i] < 0:
	    clusters[i] = -clusters[i]
print(clusters)
print(visited)
#acc = accuracy(C, y, cid+1)
#print(acc)
