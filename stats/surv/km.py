'''
Created on 27 Jun 2020

@author: snake91
'''


import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

t = 200
iterations = 100000
speed = 0.001
speedType = "const" #acc

survCurves = np.zeros((iterations, t))
hazardCurves = np.zeros((iterations, t))

pop = 100

for i in range(iterations):
    if speedType == "const":    
        x = np.cumsum(np.random.binomial(pop, speed, size = t))
    elif speedType == "acc":
        x = np.cumsum([np.random.binomial(pop, min(speed * t, 0.5), size = 1) for t in range(t)])
        
    x = np.where(x<=pop, x, np.nan)
#     x = x[pos[0]] 

    surv = 1-(x/pop)
    hazard = x/pop

    survCurves[i] = surv
    hazardCurves[i] = hazard


#     plt.plot(survCurves[i])

lbn = -1.96*np.sqrt(np.var(survCurves, axis = 0)) + np.mean(survCurves, axis = 0)
ubn =  1.96*np.sqrt(np.var(survCurves, axis = 0)) + np.mean(survCurves, axis = 0)

avg = np.mean(survCurves, axis = 0)

lb = np.percentile(survCurves, 1, axis = 0)
ub = np.percentile(survCurves, 99, axis = 0)

plt.plot(lbn)
plt.plot(ubn)
  
plt.plot(avg)
  
plt.plot(lb)
plt.plot(ub)
  
plt.show()


lbn = -1.96*np.sqrt(np.var(hazardCurves, axis = 0)) + np.mean(hazardCurves, axis = 0)
ubn =  1.96*np.sqrt(np.var(hazardCurves, axis = 0)) + np.mean(hazardCurves, axis = 0)

avg = np.mean(hazardCurves, axis = 0)

lb = np.percentile(hazardCurves, 1, axis = 0)
ub = np.percentile(hazardCurves, 99, axis = 0)


plt.plot(lbn, label = "lower bound (normal)")
plt.plot(ubn, label = "upper bound (normal)")

plt.plot(avg, label = "avg")

plt.plot(lb, label = "lower bound (%)")
plt.plot(ub, label = "upper bound (%)")

plt.legend()
plt.show()








print("")