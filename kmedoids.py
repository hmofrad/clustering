#!/usr/bin/python3.4
# Kmedoids implementation
# (c) Mohammad H. Mofrad, 2017 
# (e) hasanzadeh@cs.pitt.edu

import numpy as np
from utils import *
FILE = 'iris.data.txt'
#FILE = 'glass.data.txt'

# Read input file
# using the utils.py
[x, y] = read(FILE)

# Initliaze parameters
[n, d] = np.shape(x)  # [#samples, #dimensions]
k = len(np.unique(y)) #  #clusters


# Calculate pairwise Euclidean distance
dist = np.zeros((n,n))
for i in range(n):
   dist[i,:] = np.sqrt(np.sum(np.power(x[i,:] - x,2), axis=1))

c = np.zeros(n) # Cluster assignment
# Randomly pick k samples as
# clusters center
a = np.arange(n)
np.random.shuffle(a)
a = np.sort(a[0:k]) # Clusters center
a_t = np.copy(a)    # Copy of cluster centers

imax = 100
for i in range(imax):
   # Calculate cluster assignment 
   c = np.argmin(dist[:,a], axis=1)
   # Update cluster assignment
   for j in range(k):
      aa = np.arange(n)
      idx = aa[c == j] # Current cluster
      
      idxx = np.argmin(np.mean(dist[np.ix_(idx,idx)],axis=1))
      a_t[j] = idx[idxx]
      np.sort(a_t)
      if np.array_equal(a, a_t):
         break
      a = np.copy(a_t)
   
# Calculate accuracy
# using the utils.py
acc = accuracy(c, y, k)
print(acc)
