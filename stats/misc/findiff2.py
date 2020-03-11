'''
Created on Mar 29, 2019

@author: snake91
'''


import findiff as fd
import numpy as np
import matplotlib.pyplot as plt

def block1():
    x, y, z = [np.linspace(0, 3, 10)]*3
    dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    f = X**2 * Y * Z #np.sin(X) * np.cos(Y) * np.sin(Z)
    
    D = fd.FinDiff((0, dx, 1), (1, dy, 1), (2, dz, 1))
    res = D(f)
    
    print(res)
    
    print("")



def block2():
    x, y = [np.random.normal(size = 30)]*2
    dx, dy = 0.01, 0.01#x[1] - x[0], y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing='ij')
    f = X**2 * Y #np.sin(X) * np.cos(Y) * np.sin(Z)
    
    D = fd.FinDiff((0, dx, 1), (1, dy, 1))
    res = D(f)
    
    print(res)
    
    print("")
    
    
def block3():
    x = np.linspace(1,10,10000)
    dx  = 0.001#x[1] - x[0], y[1] - y[0]
#     X, Y = np.meshgrid(x, indexing='ij')
    f = x**2  #np.sin(X) * np.cos(Y) * np.sin(Z)
    
    D = fd.FinDiff((0, dx, 1))
    res = D(f)
    
    plt.plot(x,res)
    print(res)
    
    print("")
    
    
block3()



x = np.linspace(1,10+1/1000,1000)
y = x**2
dy = np.gradient(y)

plt.plot(x,dy)





