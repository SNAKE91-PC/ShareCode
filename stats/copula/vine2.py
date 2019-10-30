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
from multiprocessing import Pool
import scipy.optimize as spo
import scipy.integrate as scint
import functools
import pathos.pools as pp
import pandas as pd
    
# TODO: rewrite with **kwargs instead of the explicit theta
f = lambda u,v, theta: (u*v) / (u+v-u*v)

f = lambda u,v, theta: np.exp(- ( ( (-np.log(u))**theta + (-np.log(v))**theta )**(1./theta) )) #gumbel
f = lambda u,v, theta: (-1./theta) * np.log(1 + (((np.exp(-theta * u) - 1) * 
                                        (np.exp(-theta * v) - 1) )/ (np.exp(-theta) - 1)))
f = lambda u,v, theta: (u**(-theta) + v**(-theta) - 1)**(-1./theta)  #clayton



def conditionalCopula(args):
    
    v, f, theta = args
    u = np.random.uniform() # source
    g = lambda t, f, u, theta: ( f(u + 10e-7, t, theta) - f(u, t, theta) )/ 10e-7

    g = functools.partial(g, f = f, u = u, theta = theta)

    
    t = spo.brentq(lambda x: g(x) - v, 10e-7, 1-10e-7, xtol = 10e-7)
    
    return (u, t)
    
    
        

if __name__ == '__main__':
    

    v = np.random.uniform(size = 100000) #target

    pool = pp.ProcessPool(8, servers = ['localhost:9090'])
    
    
    theta = 1
    data = list(map(lambda x: tuple([x, f, theta]), v))
    copulaList = pool.map(conditionalCopula, data)
    
    x = np.array(list(map(lambda x: x[0], copulaList)))
    y = np.array(list(map(lambda x: x[1], copulaList)))
    

    
#     plt.figure()
#     plt.scatter(x,y, s=0.1)
    
    C = f(x,y,theta)    
    
    # TODO: to be fixed
    theta = 10
    data = list(map(lambda x: tuple([x, f, theta]), C))
    
    copulaList = pool.map(conditionalCopula, data)
    
    pool.close()
    
    z = np.array(list(map(lambda x: x[1], copulaList)))
    
    
    data = pd.DataFrame({'x': x, 'y': y, 'z': z})
    data.to_csv("data.csv", index = False)
    
#     plt.scatter(x,z, s = 0.1)
#     
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax = fig.gca(projection='3d')
#      
#     ax.scatter(x, y, z, s=0.75)
    
    
    
    
    
    
    
    
