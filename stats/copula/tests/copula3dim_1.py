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


from copula.copulasim import conditionalCopula2
# from copula.copulafunc import clayton

def clayton(theta, *x):
    
    if len(x)==1:
        x = x[0]
        
    return (x[0]**(-theta) + x[1]**(-theta)-1)**(-1/theta)


def clayton3d(theta, *x):
    
    if len(x)==1:
        x = x[0]
        
    return (x[0]**(-theta) + x[1]**(-theta) + x[2]**(-theta) -1)**(-1/theta)



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

    pool = pp.ProcessPool(1, servers = ['localhost:9090'])

    # F(x2|x1)
    ################################
    x1 = np.random.uniform(size = 1000) #target
    w1 = np.random.uniform(size = 1000) #target
    
    x1_w1 = tuple(zip(w1,x1))
    theta1 = 20
#     data = list(map(lambda x: list([f, theta1, x]), x1_w1))
    ftmp = functools.partial(conditionalCopula2, f = f, theta = theta1)
    copulaList = pool.map(ftmp, x1_w1)
    
    x1 = np.array(list(map(lambda x: x[0], copulaList)))
    x2 = np.array(list(map(lambda x: x[1], copulaList)))
    
    Cx1x2 = f(theta1, x1,x2)    
    
    
    ###################################
    
    
    
    print("first part")



    # F(x3|x1)    
    ################################
    theta2 = 7
    #data = list(map(lambda x: tuple([x, f, theta2]), Cx1x2))
    ftmp = functools.partial(conditionalCopula2, f = f, theta = theta2)
    
    x1_Cx1x2 = tuple(zip(Cx1x2, x1))
    
    copulaList = pool.map(ftmp, x1_Cx1x2)
    
    x1 = np.array(list(map(lambda x: x[0], copulaList)))
    x3 = np.array(list(map(lambda x: x[1], copulaList)))
    
    Cx1x3 = f(theta2, x1, x3)
    
    
    ###################################


    
    
    print("second part")

    theta3 = 20
#     v, b, f, theta
#     data = list(map(lambda x: tuple([x, f, theta3]), Cx1x3))
    ftmp = functools.partial(conditionalCopula2, f = f, theta = theta3)
    
    x1_Cx1x3 = tuple(zip(Cx1x3, x1)) 
    copulaList = pool.map(ftmp, x1_Cx1x3)
    
    x1 = np.array(list(map(lambda x: x[0], copulaList)))
    x3 = np.array(list(map(lambda x: x[1], copulaList)))
    
    Cx1x3 = f(theta3, x1,x3)

    print("third part")
    
    
    
    plt.scatter(x1,x2, s = 0.75,  color = 'blue')
    plt.scatter(x1,x3, s = 0.75,  color = 'green')
    plt.scatter(x2,x3, s = 0.75,  color = 'red')
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax = fig.gca(projection='3d')
      
    ax.scatter(x1, x2, x3, s=0.75)
    
    
    
    
    
    
    
    
