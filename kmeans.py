#!/usr/bin/python3.4
# kmeans implementation
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
[n, d] = np.shape(x)   # [#samples, #dimensions]
k = len(np.unique(y))  #  #clusters

mi = np.min(x, axis=0) # Minimum
ma = np.max(x, axis=0) # Maximum
di = ma - mi           # Difference
stop = 0               # Stopping criterion 

c = np.zeros(n)        # Cluster membership 
me = np.random.rand(k, d) * np.ones((k, d)) # Clusters mean
me = me * di
me = me + mi
me_t = np.copy(me) # Copy of clusters mean

imax = 100
for i in range(imax):
   me_t = np.copy(me)
  
   # Calculate minimum Euclidean distance and
   # update clusters membership
   for j in range(n):
      dist = np.sqrt(np.sum(np.power(x[j,:] - me,2), axis=1))
      idx = np.argmin(dist)
      val = np.min(dist)
      c[j] = idx
   
   # Calculate cluster membership and
   # update clusters mean
   for j in range(k):
      a = np.arange(n)
      idx = a[c == j] # Current cluster
      l = len(idx)    # #cluster elements
      if l:
         me[j,:] = np.sum(x[idx,:], axis=0)/len(x[idx,:])
      else:
         me[j,:] = me[j,:] + (np.random.rand(d) * di)

   # Check against stopping criterion
   st = np.sum(np.sum(np.power(me - me_t,2), axis=0))
   if(stop <= 0) or (i >= imax):
      break
# Calculate accuracy
# using the utils.py
acc = accuracy(c, y, k)
print('Accuracy: ', acc)
