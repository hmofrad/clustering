#!/usr/bin/python3.4
# Experiments 
# (c) Mohammad H. Mofrad
# (e) hasanzadeh@cs.pitt.edu

import subprocess
import numpy as np
import re
#algorithm = 'lac'
#algorithm = 'kmeans'
#algorithm = 'kmedians'
#algorithm = 'kmedoids'
algorithm = 'kmeans++'
command = 'python3.4 %s.py' %algorithm

n = 10
output = np.zeros((n,2))
print('Accuracy      Silhouette')
for i in range(n):
   temp_output = subprocess.check_output(command, shell=True)
   temp_output = str(temp_output)
   regex = re.findall(r'[-+]?\d*\.\d+|\d+', temp_output)
   output[i,:] = [regex[0], regex[1]]
   print(output[i,:])
print('Mean:', np.mean(output, axis=0))
print('Std :', np.std(output, axis=0))
