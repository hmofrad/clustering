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

def expand_cluster(points, point_index, neighbor_points, clusters, cid, epsilon, min_points):	
    [n, d] = np.shape(points)
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
clusters = -np.ones(n)        # Cluster membership 
cid = 0               # Cluster id
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

clusters = clusters - 1
print(clusters)
print(y)
estimated_num_clusters = len(np.unique(clusters))  # == cid
acc = accuracy_(clusters, y, expected_num_clusters, estimated_num_clusters)
print(acc)

#k = 3
#dist = np.zeros(n)
#for j in range(n):
#    current_point = np.tile(x[j,:],(n,1))
#    distance = np.linalg.norm(current_point - x, axis=1)
#    distance = np.sort(distance)
#    dist[j] = distance[k]


#for j in range(n):
#    print(kdist[j])
#mean = np.mean(kdist)
#std = np.std(kdist)
#print(mean, '+/-', std)	

'''
anchor = mean + (std * std_away)
print('anchor: ', anchor)
last = None
for kd in reversed(kdist):
    #print(kd)
    if kd > anchor:
        last = kd
        print(last)
    if kd < anchor and last is not None:
        print('kdist:', last)
        break
        #return last
if(last == None):
    raise Exception('sd too far away from mean')
#print('kdist:', last)		
'''  
