#!/usr/bin/python3.4
# DBSCAN implementation
# (c) Mohammad HMofrad, 2017 
# (e) mohammad.hmofrad@pitt.edu

import numpy as np
from utils import *

def region_query(points, point_index, epsilon):
    [n, d] = np.shape(points)
    current_point = np.tile(points[point_index,:],(n,1))
    distance = np.linalg.norm(current_point - points, axis=1)
    p = np.arange(n)
    neighbor_points = p[(distance <= epsilon)]
    return neighbor_points

def expand_cluster(points, point_index, neighbor_points, clusters, cluster_id, epsilon, min_points):	
    [n, d] = np.shape(points)
    clusters[point_index] = cluster_id
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
            clusters[point] = cluster_id

def k_distance(points, k, std_away):
    [n, d] = np.shape(points)
    k_dist = np.zeros(n)
    for j in range(n):
        current_point = np.tile(points[j,:],(n,1))
        dist = np.linalg.norm(current_point - points, axis=1)
        dist = np.sort(dist)
        k_dist[j] = dist[k]	
    k_dist = np.sort(k_dist)		

    mean = np.mean(k_dist)
    std = np.std(k_dist)
    print(mean, '+/-', std)

    anchor = mean + (std * std_away)
    print('anchor:', anchor)
    last = None
    for kd in reversed(k_dist):
        if kd > anchor:
            last = kd
        if kd < anchor and last is not None:
            print('k-dist:', last)
            return last
    if(last == None):
        raise Exception('std is too far away from mean')	
	
def clculate_centroids(points, clusters, k):
    [n, d] = np.shape(points)
    me = np.zeros((k, d))
    for j in range(k):
       a = np.arange(n)
       idx = a[clusters == j]
       l = len(idx)
       if l:
          me[j,:] = np.sum(points[idx,:], axis=0)/len(points[idx,:])
       else:
          me[j,:] = me[j,:] + (np.random.rand(d))	
    return(me)
	
#    return(k_distance)		
		
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
clusters = -np.ones(n)       # Cluster membership 
cluster_id = 0               # Cluster id
expected_num_clusters = len(np.unique(y))  #clusters
visited = np.zeros(n)  # Visited

min_points = d
k = min_points
for std_away in range(5):
    try:
        k_dist = k_distance(x, k, std_away)
        break
    except:
        print('k-dist anchor is out of range, skipping')
        k_dist = 1
        continue
epsilon = k_dist

starting_point = np.random.randint(0,n, 1)[0]
for i in range(starting_point, starting_point + n):
    j = i - starting_point
#    print(j)
    if(visited[j] == 1):
        pass
    else:
        visited[j] = 1
        neighbor_points = region_query(x, j, epsilon)
        if(neighbor_points.size < min_points):
            clusters[j] = -cluster_id
        else: # Expand the cluster
            cluster_id = cluster_id + 1
            expand_cluster(x, j, neighbor_points, clusters, cluster_id, epsilon, min_points)

for i in range(n):
    if clusters[i] < 0:
	    clusters[i] = -clusters[i]

clusters = clusters - 1
print(clusters)
print(y)
estimated_num_clusters = len(np.unique(clusters))  # == cluster_id
acc = accuracy_(clusters, y, expected_num_clusters, estimated_num_clusters)
me = clculate_centroids(x, clusters, k)
sil = silhouette(x, clusters, me)
print(acc, sil)


