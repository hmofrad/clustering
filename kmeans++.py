#!/usr/bin/python3.4
# kmeans implementation
# (c) Mohammad H. Mofrad, 2017 
# (e) hasanzadeh@cs.pitt.edu

import numpy as np
from utils import *
import scipy

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

c1 = round(np.random.uniform(0,n))
D2 = np.zeros(n)
C = [c1] #initial mean
print(C)
X = [0, 1, 2, 3, 4]
for kk in range(1,k):
    for i in range(0, n):
#    print(x[i,:] - x[c1,:])
#    print(np.power(x[i,:] - x[c1,:],2))
    
#    for c in C:
#        print(x[i,:])
#        print(x[c,:])
#        print(np.sqrt(np.sum(np.power(x[i,:] - c, 2))))
#        print(np.power(np.sum(x[i,:] - c), 2))
        D2[i] = min([np.power(np.sum(x[i,:] - c), 2) for c in C])
    probs = D2/D2.sum()
    cumprobs = probs.cumsum()
    r = np.random.uniform(0,1)
    idx = np.where(cumprobs <= r)[0]
    if not idx.size:
        idx = 0
    else:
        idx = max(idx)
    C.append(idx)
#    print(r)
  
#    print(np.where(cumprobs <= r)[0])
#    midx = max(np.where(cumprobs <= r)[0])
#    exit(0)
#    idx = np.argmin(dist)
#    val = np.min(dist)
#    c[i] = idx
#    print()
#print(D2)
#print(cumprobs)
print('--->',x[C,:])
me = x[C,:] # Clusters center
me_t = np.copy(me) # Copy of clusters center
#print(me)
#print(np.random.uniform(0,1))
#print(me)
#dx = np.sqrt(np.sum(np.power(x[0,:] - x[c1,:],2), axis=1))
#print(dx)
#ii = 0
#r = scipy.rand()
r = 0.004
print(idx)

#print(r, n)
#for j,p in enumerate(cumprobs):
#    print(j,p)
#    if r <= p:
#        print('--->', j,p)
#        ii = j
#        break
#print(ii)
#pp = np.where(cumprobs <= r)[0]
#print(max(pp))
#print(cumprobs)

def initialize(X, K):
    C = [X[0]]
    for k in range(1, K):
        D2 = scipy.array([min([scipy.inner(c-x,c-x) for c in C]) for x in X])
        for x in X:
            for c in C:
                print(x, c, c - x, scipy.inner(c-x,c-x))
        print('-->', D2, D2/D2.sum(), D2/sum(D2))
        probs = D2/D2.sum()
        cumprobs = probs.cumsum()
        print(cumprobs)
        r = scipy.rand()
        for j,p in enumerate(cumprobs):
            if r < p:
                i = j
                break
        C.append(X[i])
    return C

#print(initialize(X, 3))

#exit(0)
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
         me[j,:] = np.mean(x[idx,:], axis=0)
      else:
         me[j,:] = me[j,:] + (np.random.rand(d) * di)

   # Check against stopping criterion
   st = np.sum(np.sum(np.power(me - me_t,2), axis=0))
   if(stop <= 0) or (i >= imax):
      break

# Calculate accuracy and
# Silhouette Coefficient
# using utils.py
acc = accuracy(c, y, k)
sil = silhouette(x, c, me)
print(acc, sil)
