#!/usr/bin/python3.4
# Kmedoids implementation
# (c) Mohammad H. Mofrad, 2017 
# (e) hasanzadeh@cs.pitt.edu

import numpy as np
from utils import *
np.random.seed()

# Read and store the input data
# using the utils.py
PERFIX = 'dataset/'
#FILE = PERFIX + 'balance-scale.data.txt'
#FILE = PERFIX + 'breast-cancer-wisconsin.data.txt'
#FILE = PERFIX + 'sonar.all-data.txt'
FILE = PERFIX + 'cmc.data.txt'
#FILE = PERFIX + 'glass.data.txt'
#FILE = PERFIX + 'hayes-roth.data.txt'
#FILE = PERFIX + 'ionosphere.data.txt'
#FILE = PERFIX + 'iris.data.txt'
#FILE = PERFIX + 'pima-indians-diabetes.data.txt'
#FILE = PERFIX + 'wine.data.txt'

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
   
# Calculate accuracy and
# Silhouette Coefficient
# using the utils.py
acc = accuracy(c, y, k)

me = np.zeros((k,d))
for j in range(k):
   a = np.arange(n)
   idx = a[c == j]
   l = len(idx)
   if l:
      me[j,:] = np.sum(x[idx,:], axis=0)/len(x[idx,:])
   else:
      me[j,:] = me[j,:] + (np.random.rand(d))

sil = silhouette(x, c, me)
print(acc, sil)
