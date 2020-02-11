'''
Created on 30 Jan 2020

@author: snake91
'''


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


f = lambda u,v, theta: (u*v) / (u+v-u*v)

f = lambda u,v, theta: np.exp(- ( ( (-np.log(u))**theta + (-np.log(v))**theta )**(1./theta) )) #gumbel
f = lambda u,v, theta: (-1./theta) * np.log(1 + (((np.exp(-theta * u) - 1) * 
                                        (np.exp(-theta * v) - 1) )/ (np.exp(-theta) - 1)))
f = lambda u,v, theta: (u**(-theta) + v**(-theta) - 1)**(-1./theta)  #clayton


def copulamle(theta, f, u,v,C):
    
    x = f(u,v,theta)
    x = np.where(x < 0, 0, x)
    x = np.where(x > 1, 1, x)
    
    res = np.log(np.mean((x - C)**2))
    
    print(theta[0], res, sep = " ")
    return res




if __name__ == "__main__":
    
    
    import pandas as pd
    import scipy.optimize as spo
    
    data = pd.read_csv("/home/snake91/data3d.csv")

    x = data['x']
    y = data['y']
    z = data['z']
    Cxy = data['Cxy']
    Cxy_z = data['Cxy_z']

#     print(copulamle(50, x, y, Cxy))
    
    args = tuple(x,y,Cxy)
    paramUV = spo.minimize(copulamle, x0 = (2,), bounds = [(0, None)], args = (f, args))
    print(paramUV)
    
    args = Cxy,z,Cxy_z
    paramVZ = spo.minimize(copulamle, x0 = (2,), bounds = [(0, None)], args = (f, args))
    print(paramVZ)

    print("")
    
    plt.scatter(x[:10000],y[:10000], s= 0.25, color = 'red')
    plt.scatter(Cxy[:10000],z[:10000], s= 0.25, color = 'blue')
    
    ''' C(C(x,y),z) '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax = fig.gca(projection='3d')
      
    ax.scatter(x[:10000], y[:10000], z[:10000], s=0.75)
    
    
    
    
    
    
    
    