'''
Created on Dec 13, 2018

@author: snake91
'''

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import pathos.pools as pp
import pandas as pd
    
    
from copula.copulasim import conditionalCopula1
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

    def clayton(theta, *x):
    
        if len(x)==1:
            x = x[0]
            
        return (x[0]**(-theta) + x[1]**(-theta)-1)**(-1/theta)

    f = clayton
    v = np.random.uniform(size = 10000) #target

    pool = pp.ProcessPool(4)
    
    theta = 10

    data = list(map(lambda x: tuple([x, f, theta]), v))
    copulaList = pool.map(conditionalCopula1, data)
    
    x = np.array(list(map(lambda x: x[0], copulaList)))
    y = np.array(list(map(lambda x: x[1], copulaList)))
    
    xsample = x
    ysample = y
#     plt.figure()
#     plt.scatter(x,y, s=0.1)
    
    C = f(x,y,theta)    
    
    plt.scatter(st.norm.ppf(x),st.norm.ppf(y), s = 0.7)

    data = pd.DataFrame({'x': x, 'y': y, 'C': C})
    data.to_csv("/home/snake91/data.csv", index = False)
    
    print("done")
#     plt.scatter(x,z, s = 0.1)
#     
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax = fig.gca(projection='3d')
#      
#     ax.scatter(x, y, z, s=0.75)
    
    
    
    
    
    
    
    
