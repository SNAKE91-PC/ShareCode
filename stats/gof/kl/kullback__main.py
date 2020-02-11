'''
Created on 26 Dec 2019

@author: snake91
'''


import os
import sys

import datetime
import subprocess

# lst = ['python3', '/home/snake91/git/ShareCode/stats/gof/setup.py', 'clean', '--all', 'build_ext']
# subprocess.call(lst, shell = True)

os.system('python3 /home/snake91/git/ShareCode/stats/gof/kl/setup.py clean --all build_ext')

sys.path.append("/home/snake91/git/ShareCode/stats/gof/kl/build/lib.linux-x86_64-3.7/gof/kl")

# os.chdir("/home/snake91/git/ShareCode/stats/gof/build/lib.linux-x86_64-3.7/gof")
import kullback as klcy #@unresolvedimport
 
import datetime
import numpy as np
import matplotlib.pyplot as plt
 
import pathos.pools as pp
import objgraph
 
np.random.seed(10)
  
ncore = 4
pool = pp.ProcessPool(ncore)

KL = []

for i in range(2500):
 
    time1 = datetime.datetime.now()
      
    n = 500                                           
    x = [np.random.normal(size = n, scale = 1) for j in range(ncore)]
    y = [np.random.normal(size = n, scale = 1) for j in range(ncore)]

    data = np.array(list(zip(x,y,[n/10]*ncore)))
    
    kl = pool.map(klcy.KullbackLeibler, data)
   
#     kl = klcy.KullbackLeibler(x, y, int(n/10))
 
    time2 = datetime.datetime.now()
  
    print(i, time2 - time1, sep = "      ")
  
#     print(objgraph.show_growth())
    KL.append(kl)


flatten = lambda l: [item for sublist in l for item in sublist]

KL = flatten(KL)
plt.hist(KL, bins = 500)

plt.savefig("/home/snake91/git/ShareCode/stats/gof/pics/" + "KL.svg")





