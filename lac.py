#!/usr/bin/python3.4
# Learning Automata Clusteing (LAC) implementation
# (c) Mohammad HMofrad, 2016
# (e) mohammad.hmofrad@pitt.edu

import numpy as np
from utils import *
np.random.seed()

# Read and store the input data
# using the utils.py
PREFIX = 'dataset/'
#FILE = PREFIX + 'balance-scale.data.txt'
#FILE = PREFIX + 'breast-cancer-wisconsin.data.txt'
#FILE = PREFIX + 'sonar.all-data.txt'
#FILE = PREFIX + 'cmc.data.txt'
#FILE = PREFIX + 'glass.data.txt'
#FILE = PREFIX + 'hayes-roth.data.txt'
#FILE = PREFIX + 'ionosphere.data.txt'
#FILE = PREFIX + 'iris.data.txt'
#FILE = PREFIX + 'pima-indians-diabetes.data.txt'
FILE = PREFIX + 'wine.data.txt'
#FILE = PREFIX + 'drift.data.txt'
#FILE = PREFIX + 'har.data.txt'
#FILE = PREFIX + 'soybean-small.data.txt'
#FILE = PREFIX + 'segmentation.data.txt'
[x, y] = read(FILE)

# Initliaze parameters
[n, d] = np.shape(x)
k = len(np.unique(y))

numactions = k
alpha = 0.45
beta = 0.09
#alpha = 0.9
#beta = 0.01

action = np.zeros((k, n)) # LA action set
probability = np.tile(1/numactions, (numactions, n)) # LA prob. set

mi = np.min(x, axis=0) # Minimum
ma = np.max(x, axis=0) # Maximum
di = ma - mi           # Difference

c = np.zeros(n)        # LA cluster membership
cc = np.zeros(n)       # RS cluster membership
me = np.random.rand(k, d) * np.ones((k, d)) # Clusters mean
me = me * di
me = me + mi
me_t = np.zeros(np.shape(me)) # Copy of clusters mean

imax = 100
for i in range(imax):
   # Select an action based on prob.
   # update clusters membership 
   action = actionselection(action, probability, numactions, n)
   signal   = np.ones(n)
   for j in range(k):
      a = np.arange(n)
      c[action[j,:] == 1] = j

   me_t = np.copy(me)
   # Calculate minimum Euclidean distance and
   # update kmeans clusters membership 
   for j in range(n):
      dist = np.sqrt(np.sum(np.power(x[j,:] - me,2), axis=1))
      idx = np.argmin(dist)
      val = np.min(dist)
      cc[j] = idx

   # Calculate kmeans cluster membership and
   # update kmeans clusters mean
   for j in range(k):
      a = np.arange(n)
      idx = a[c == j] # Current cluster
      l = len(idx)    # #cluster elements
      if l:
         me[j,:] = np.mean(x[idx,:], axis=0)
   
   # Compute reinforcement signal   
   for j in range(n):
      if c[j] == cc[j]:
         signal[j] = 0
   # Update probability vector
   probability = probabilityupdate(action, probability, numactions, n, signal, alpha, beta)

   # Check against stopping criterion
   stop = np.sum(np.sum(np.power(me - me_t,2), axis=0))
   if(stop <= 0) or (i >= imax):
      break

# Calculate accuracy and
# Silhouette Coefficient
# using the utils.py
acc = accuracy(c, y, k)
sil = silhouette(x, c, me)
print(acc, sil)

# New experiment
#print(c)
#print(y)
#uniq = np.unique(y)
 
#for u in uniq:
#    print('Cluster', u + 1, np.unique(c[y == u]) + 1)