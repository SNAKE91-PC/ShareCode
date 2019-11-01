'''
Created on Oct 27, 2019

@author: snake91
'''

import numpy as np
import matplotlib.pyplot as plt



n_size   = 100
n_series = 1000000
x = [np.random.uniform(size = n_size) for i in range(n_series)]

def onesample_theor_ks(x):
 
    u = np.linspace(0,1, len(x))
    x = np.sort(x)
     
    diff = np.max(np.abs(x - u))
         
 
    return diff


def twosample_emp_ks(x, y):
    
    x = np.sort(x)
    y = np.sort(y)

    diff = np.max(np.abs(x-y))
    
    return diff

def nsample_theor_ks(x):

    u = np.linspace(0,1, len(x[0]))
    x = list(x)
    
    maxList = []
    for i in range(len(x)):
        x[i] = np.sort(x[i])
    
        diff = np.max(np.abs(x[i] - u))
    
        maxList.append(diff)
        
    
    return np.max(maxList)


# onesample_theor_statsks = np.array([onesample_theor_ks(x[i]) for i in range(len(x)) ])
# twosample_emp_statsks = np.array([twosample_emp_ks(x[i], x[i-1]) for i in range(1,len(x))])

# sample_theor_statsks1 = np.array([nsample_theor_ks(x[i]) for i in range(0,len(x))])
# sample_theor_statsks2 = np.array([nsample_theor_ks(x[i], x[i-1]) for i in range(1,len(x))])
# sample_theor_statsks3 = np.array([nsample_theor_ks(x[i], x[i-1], x[i-2]) for i in range(2,len(x))])
# sample_theor_statsks4 = np.array([nsample_theor_ks(x[i], x[i-1], x[i-2], x[i-3]) for i in range(3,len(x))])
# 
# 
# plt.hist(sample_theor_statsks1, bins = 300, histtype='step', label = 'one sample theor')
# plt.hist(sample_theor_statsks2, bins = 300, histtype='step', label = 'two sample emp')
# plt.hist(sample_theor_statsks3, bins = 300, histtype='step', label = 'three sample emp')
# plt.hist(sample_theor_statsks4, bins = 300, histtype='step', label = 'four sample emp')

maxid = 10
statsList = []
for i in range(1, maxid):

    sample_theor_statsks = [nsample_theor_ks(x[j-i: j]) for j in range(i, len(x) - (maxid - i))]
    statsList.append(sample_theor_statsks)
    

for i in range(len(statsList)):
    
    plt.hist(statsList[i], bins = 400, histtype='step', label = 'sample theor ' + str(i))
    
plt.legend()

plt.show()














