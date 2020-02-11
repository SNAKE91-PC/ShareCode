'''
Created on Apr 15, 2019

@author: snake91
'''

'''
Created on Dec 13, 2018

@author: snake91
'''

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import scipy.optimize as spo
import functools
import pathos.pools as pp
import pandas as pd


from copula.copulasim import conditionalCopula
from copula.copulafunc import clayton

# TODO: rewrite with **kwargs instead of the explicit theta
# f = lambda u,v, theta: (u*v) / (u+v-u*v)
# 
# f = lambda u,v, theta: np.exp(- ( ( (-np.log(u))**theta + (-np.log(v))**theta )**(1./theta) )) #gumbel
# f = lambda u,v, theta: (-1./theta) * np.log(1 + (((np.exp(-theta * u) - 1) * 
#                                         (np.exp(-theta * v) - 1) )/ (np.exp(-theta) - 1)))
# f = lambda u,v, theta: (u**(-theta) + v**(-theta) - 1)**(-1./theta)  #clayton



if __name__ == '__main__':
    
    f = clayton
    
    np.random.seed(100)

    v = np.random.uniform(size = 10000) #target

    pool = pp.ProcessPool(1, servers = ['localhost:9090'])
    
    theta = 5
    data = list(map(lambda x: tuple([x, f, theta]), v))
    copulaList = pool.map(conditionalCopula, data)
    
    x = np.array(list(map(lambda x: x[0], copulaList)))
    y = np.array(list(map(lambda x: x[1], copulaList)))
    
    
    Cxy = f(x,y,theta)    
    
    print("first part")
    
    theta = 50
    data = list(map(lambda x: tuple([x, f, theta]), Cxy))
    
    copulaList = pool.map(conditionalCopula, data)
    
    pool.close()
    
    z = np.array(list(map(lambda x: x[1], copulaList)))
    
    Cxy_z = f(Cxy, z, theta) 
    data = pd.DataFrame({'x': x, 'y': y, 'z': z, 'Cxy': Cxy,'Cxy_z': Cxy_z}) # C(u,C(v,z))
    data.to_csv("/home/snake91/data3d.csv", index = False)
    
    print("done")
#     

    plt.scatter(x,y, s = 0.75,  color = 'blue')
    plt.scatter(Cxy, z, s =0.75, color = 'red')
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax = fig.gca(projection='3d')
      
    ax.scatter(x, y, z, s=0.75)
    
    
    
    
    
    
    
    
