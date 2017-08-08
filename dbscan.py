#!/usr/bin/python3.4
# DBSCAN implementation
# (c) Mohammad HMofrad, 2017 
# (e) mohammad.hmofrad@pitt.edu

import numpy as np
from utils import *

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
print(np.shape(x))

# Initliaze parameters
[n, d] = np.shape(x)   # [#samples, #dimensions]
#k = len(np.unique(y))  #  #clusters
epsilon = 1
min_points = d
starting_point = np.random.randint(0,n, 1)[0]
print(epsilon)
print(starting_point)
