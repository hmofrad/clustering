#!/usr/bin/python3.4
# DBLAC implementation
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
	
def expand_actions(actions):
    [num_actions, n] = np.shape(actions)
    print(num_actions, n)
    num_actions = num_actions + 1
    actions = np.append(actions,np.zeros((1, n)), axis=0)
    return actions, num_actions

def expand_probabilities(probabilities):
    [num_probs, n] = np.shape(probabilities)
    num_probs = num_probs + 1
    probabilities = np.append(probabilities,np.zeros((1, n)), axis=0)
    probabilities = probabilities - (1/num_actions * probabilities)
    probabilities[-1,:] = 1/num_actions
    return probabilities, num_probs

def expand_la(actions, probabilities):
    actions, num_actions = expand_actions(actions)
    probabilities, num_probs = expand_probabilities(probabilities)
    return(actions, num_actions, probabilities)

	
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
n = 10
num_actions = 1
actions = np.zeros((num_actions, n)) # LA action set
probabilities = np.tile(1/num_actions, (num_actions, n)) # LA probability set
alpha = 0.45
beta = 0.09

print(actions)
print(probabilities)
#action[0,0] = 1
#action = np.append(action,np.zeros((num_actions, n)))
#print(action)

actions = actionselection(actions, probabilities, num_actions, n)

#actions, num_actions = expand_actions(actions)
#print(actions)


signal   = np.ones(n)
for j in range(n):
    if(np.random.rand() > 0.5):
        signal[j] = 0
print(signal)

probabilities = probabilityupdate(actions, probabilities, num_actions, n, signal, alpha, beta)
print(probabilities)

actions, num_actions, probabilities = expand_la(actions, probabilities)

#actions, num_actions = expand_actions(actions)
print(actions)

#probabilities = expand_probabilities(probabilities)
print(probabilities)
print(num_actions)
actions = actionselection(actions, probabilities, num_actions, n)
print(actions)


probabilities = probabilityupdate(actions, probabilities, num_actions, n, signal, alpha, beta)
print(probabilities)



exit(0)


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


