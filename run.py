#!/usr/bin/python3.4
# Experiments 
# (c) Mohammad H. Mofrad
# (e) hasanzadeh@cs.pitt.edu

import subprocess
import numpy as np

algorithm = 'lac'
#algorithm = 'kmeans'
#algorithm = 'kmedoids'

command = 'python3.4 %s.py' %algorithm

n = 10
output = np.zeros(n)
for i in range(n):
   temp_output = subprocess.check_output(command, shell=True)
   temp_output = float(temp_output)
   output[i] = temp_output
   print("accuracy:", temp_output)
print(np.mean(output), '+/-', np.std(output))
