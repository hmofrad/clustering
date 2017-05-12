#!/usr/bin/python3.4
# Learning Automata Clusteing (LAC) profiling script
# (c) Mohammad H. Mofrad, 2016
# (e) mohammad.hmofrad@pitt.edu

import numpy as np
from utils import *

# Read and store the input data
FILE = 'iris.data.txt'
#FILE = 'glass.data.txt'
# Read input file
# using the utils.py
[x, y] = read(FILE)
f=open('foo.csv','wb')
f1=open('foo1.csv','wb')
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
stop = 0               # Stopping criterion

c  = np.zeros(n)        # LA cluster membership
cc = np.zeros(n)       # RS cluster membership
me = np.random.rand(k, d) * np.ones((k, d)) # Clusters mean
me = me * di
me = me + mi
me_t = np.zeros(np.shape(me)) # Copy of clusters mean
imax = 100
ppm = -1 + np.zeros((k,imax))
pps = -1 + np.zeros((k,imax))
for i in range(imax):
   # Select an action based on prob.
   # update clusters membership 
   action = actionselection(action, probability, numactions, n)
   signal   = np.ones(n)
   for j in range(k):
      a = np.arange(n)
      c[action[j,:] == 1] = j
   for j in range(k):
  #    print(c == j)
 #     print(probability[:,c == j])
      ppm[j,i] = np.mean(np.max(probability[:,c == j], axis=0))
      pps[j,i] = np.std(np.max(probability[:,c == j], axis=0))
   print(ppm[:,i])
   foo = np.array([ppm[:,i]])
   foo1 = np.array([pps[:,i]])
#   print(foo)
#   exit(0)
   np.savetxt(f, foo, fmt='%1.9f')
   np.savetxt(f1, foo1, fmt='%1.9f')
  #    print(np.mean(np.max(probability[:,c == j], axis=0)), np.std(np.max(probability[:,c == j], axis=0)))
     # print(ppm[j,i])
#print(np.mean(np.max(probability, axis=0)), np.std(np.max(probability, axis=0)))
#print(np.argmax(probability, axis=0)
 #  exit(0)
   me_t = np.copy(me)
   # Calculate minimum Euclidean distance and
   # update kmeands clusters membership 
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
         #me[j,:] = np.sum(x[idx,:], axis=0)/len(x[idx,:])
         me[j,:] = np.mean(x[idx,:], axis=0)
#      else:
#         me[j,:] = me[j,:] + (np.random.rand(d) * di)
   
   # Compute reinforcement signal   
   for j in range(n):
      if c[j] == cc[j]:
         signal[j] = 0
   # Update probability vector
#   foo = np.array([probability[:,1]])
#   print(foo)
#   np.savetxt(f, foo, fmt='%1.9f')
#   print(probability[:,1])
   
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
f.close()
f1.close()
print(ppm)
print(pps)
#print(np.mean(np.max(probability, axis=0)), np.std(np.max(probability, axis=0)))
#print(np.argmax(probability, axis=0))
