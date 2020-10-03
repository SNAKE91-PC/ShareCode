'''
Created on Dec 13, 2018

@author: snake91
'''

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import pathos.pools as pp
import pandas as pd
    
    
from copula.copulasim import conditionalCopula2
from copula.copulafunc import clayton

# from copula.copulafunc import clayton
# TODO: rewrite with **kwargs instead of the explicit theta
# f = lambda u,v, theta: (u*v) / (u+v-u*v)
# 
# f = lambda u,v, theta: np.exp(- ( ( (-np.log(u))**theta + (-np.log(v))**theta )**(1./theta) )) #gumbel
# f = lambda u,v, theta: (-1./theta) * np.log(1 + (((np.exp(-theta * u) - 1) * 
#                                         (np.exp(-theta * v) - 1) )/ (np.exp(-theta) - 1)))
# f = lambda u,v, theta: (u**(-theta) + v**(-theta) - 1)**(-1./theta)  #clayton


        


if __name__ == '__main__':
    
    import scipy.stats as st

    f = clayton
    
    np.random.seed(10)
    
    v1 = np.random.uniform(size = 10000) #v1 ---> v2
    q =  np.random.uniform(size = 10000) #quantile

    pool = pp.ProcessPool(16)
    
    theta = 10

    pairsvq = list(zip(list(v1), list(q)))
    data = list(map(lambda x: tuple([x, f, theta]), pairsvq))
    copulaList = pool.map(conditionalCopula2, data)
    
    pool.close()
    pool.clear()
    
    xy = np.array(list(map(lambda x: x[0], copulaList)))
    q = np.array(list(map(lambda x: x[1], copulaList)))
    q = np.reshape(q, (q.shape[0],1))

    idxNones = [i for i in range(xy.shape[0]) if xy[i,1] is None or np.isnan(xy[i,1])] 
    xyq = np.hstack([xy, q])
    xyqNones = xyq[idxNones]


    select = np.in1d(range(xyq.shape[0]), idxNones)
    xyq = xyq[~select]
    
#     plt.scatter(list(map(lambda x: x[0], xyqNones)), list(map(lambda x: x[2], xyqNones)))


    xsample = list(map(lambda x: x[0], xyq))
    ysample = list(map(lambda x: x[1], xyq))
    qsample = list(map(lambda x: x[2], xyq))
#     plt.figure()
#     plt.scatter(x,y, s=0.1)

    
    C = list(map(lambda x: f(theta, x[0], x[1]), list(zip(xsample, ysample))))
    
#     plt.scatter(st.norm.ppf(xsample),st.norm.ppf(ysample), s = 0.7)

    data = pd.DataFrame({'x': xsample, 'y': ysample, 'C': C})
    
#     data.to_csv("/home/snake91/data.csv", index = False)
    
    print("done 2dim")
    
    z = np.array(C)
    pairszq = list(zip(list(z), list(q))) 
    
    pool = pp.ProcessPool(16)
    
    theta2 = 5
    data = list(map(lambda x: tuple([x, f, theta2]), pairszq))
    copulaList = pool.map(conditionalCopula2, data)
    
    xy = np.array(list(map(lambda x: x[0], copulaList))) #x = C
    q = np.array(list(map(lambda x: x[1], copulaList)))  
    q = np.reshape(q, (q.shape[0],1))
    
    u3sample = np.array(list(map(lambda x: x[1], xy)), dtype = np.float64)
    u1sample = np.array(xsample)
    u2sample = np.array(ysample)
    
    idxNones = [i for i in range(u3sample.shape[0]) if u3sample[i] is None or np.isnan(u3sample[i])]
    select = np.in1d(range(u3sample.shape[0]), idxNones)
    u3sample = u3sample[~select]
    u2sample = u2sample[~select]
    u1sample = u1sample[~select]
    
    pool.clear()
    pool.close()
    
    plt.scatter(u1sample, u2sample, s = 0.5)
    plt.scatter(u1sample, u3sample, s = 0.5)
    plt.scatter(u2sample, u3sample, s = 0.5)
    
    print("")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#     plt.scatter(x,z, s = 0.1)
#     
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax = fig.gca(projection='3d')
#      
#     ax.scatter(x, y, z, s=0.75)
    
    
    
    
    
    
    
    
