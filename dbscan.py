#!/usr/bin/python3.4
# DBSCAN implementation
# (c) Mohammad HMofrad, 2017 
# (e) mohammad.hmofrad@pitt.edu



import numpy as np
from utils import *


def region_query(points, point, epsilon):
    current_point = np.tile(point,(n,1))
    distance = np.linalg.norm(current_point - points, axis=1)
    p = np.arange(n)
    neighbor_points = p[(distance <= epsilon)]
    return neighbor_points

def expand_cluster(points, neighbor_points, clusters, epsilon, min_points):	
    pass
	
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

min_points = d
starting_point = np.random.randint(0,n, 1)[0]
#for j in range(n):
#    Neighbor_Points = []
    #print(np.sqrt(np.sum(np.power(x[j,:] - x[j,:],2))))
#    dist = np.sqrt(np.sum(np.power(x[starting_point,:] - x[j,:],2)))
#    if dist < epsilon:
#	    Neighbor_Points.append(j)
#	    print(j)
		
#print(Neighbor_Points)
#print(np.tile(x[starting_point,:], (n,1)))
#for j in range(n):
#visited[149] = 1
for j in range(n):
   # print(j, visited[j])
    if(visited[j] == 1):
        pass
    else:
        visited[j] = 1
        neighbor_points = region_query(x, x[j,:], epsilon)
        #current_point = np.tile(x[j,:],(n,1))
        #distance = np.linalg.norm(current_point - x, axis=1)
        #p = np.arange(n)
        #neighbor_points = p[(distance <= epsilon)]
        if(neighbor_points.size < min_points):
            clusters[j] = -cid
        else: # Expand the cluster
            cid = cid + 1
            clusters[j] = cid
            #print(neighbor_points)
            k = 0
            while(True):
            #for point in neighbor_points:
                point = neighbor_points[k]
                #print(k, point, neighbor_points.size)
                if(visited[point] == 0):
                    visited[point] = 1
                    neighbor_points_ = region_query(x, x[point,:], epsilon)
                    #current_point = np.tile(x[point,:],(n,1))
                    #distance = np.linalg.norm(current_point - x, axis=1)
                    #p = np.arange(n)
                    #neighbor_points_ = p[(distance <= epsilon)]                
                    if(neighbor_points_.size >= min_points):
                        neighbor_points = np.append(neighbor_points, np.setdiff1d(neighbor_points_,neighbor_points))
                        #neighbor_points = np.array(neighbor_points)
                        #print(np.setdiff1d(neighbor_points_,neighbor_points))
                        #pass
                        #print('ADD', neighbor_points)
                k = k + 1
                if(k == neighbor_points.size):
                    break
                if(clusters[point] < 0):
                    clusters[point] = cid

#print('1', neighbor_points)
#print('2', neighbor_points_)

#set_diff = np.setdiff1d(neighbor_points_,neighbor_points)
#print(set_diff)			
		
#neighbor_points = np.append(neighbor_points, 15)

#print('2', neighbor_points)
#neighbor_points = np.append(neighbor_points, 100)
#print('1', neighbor_points)
#print('2', neighbor_points_)
#neighbor_points = np.sort(neighbor_points)
#neighbor_points_ = np.sort(neighbor_points_)
#print('1', np.setdiff1d(neighbor_points, neighbor_points_))
#print('2', np.setdiff1d(neighbor_points_,neighbor_points))
#print(np.digitize(neighbor_points, neighbor_points_))
#print(np.digitize(neighbor_points_, neighbor_points))
	#if
    #print(j, neighbor_points.size)
#print(idx, idx.size)
#C[idx] = cid;
#cid = cid + 1;
print(clusters)
#print(y)
#print(np.size(C == 3), d)
#s = 0
for i in range(n):
    if clusters[i] < 0:
	    clusters[i] = -clusters[i]
print(clusters)
#print(s)
#print(visited)
#visited[idx[0]] = 1
#print('---', idx[0])
#print(p[(dist <= epsilon)])
#print(p[(visited == 0)])
#print('+++')
#print(p[(visited == 0) - (dist <= epsilon)])
#print(p[(dist <= epsilon) - (visited == 0)])

#print(p[(visited == 0)] - p[(dist <= epsilon)])
#print(p[(dist <= epsilon)] - p[(visited == 0)])


#print(dist[0:-1], np.shape(dist[0:]))
#print(dist[0])
    #print(dist, dist < epsilon)
#for i in range(n):
#	print(i, visited[i])
#	visited[i] = 0


#if(!visited)
#print(cid)
#acc = accuracy(C, y, cid+1)
#print(acc)
