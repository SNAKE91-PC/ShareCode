'''
Created on 12 Feb 2020

@author: snake91
'''

import numpy as np
from copy import deepcopy
import functools
import scipy.special as spe
from copy import deepcopy
# first derivative one variable

h = 0.01

"""
    f univariate
    g bivariate
    t trivariate
"""


def f(x):
    
    return x**5

def g(x,y):
    
    return x**3 * y**2

def t(x,y,z):

    return x**2 *y *z



def df(f,u):

    return (f(u + h) - f(u)) / h

def ddf_1(f,u):

    tmp1 = ( (f(u  + 2 * h) - f(u + 1 * h)) / h ) # df/du (u=u+h)
    tmp2 = ( (f(u +  1 * h) - f(u + 0 * h)) / h ) # df/du (u=u  )
             
    return (tmp1 - tmp2) /h
             
def ddf_2(f,u):
    
    return (df(f,u + h) - df(f,u))/h

    
def dddf_1(f,u):
    
    tmp1_1 =      (( f(u  + 3 * h) - f(u + 2 * h)) / h ) 
    tmp1_2 =      (( f(u +  2 * h) - f(u + 1 * h)) / h )
    
    tmp2_1 = (tmp1_1 - tmp1_2) / h # d2f/du2 (u=u+h)
    
    tmp1_3 =      (( f(u  + 2 * h) - f(u + 1 * h)) / h )
    tmp1_4 =      (( f(u  + 1 * h) - f(u + 0 * h)) / h )
    
    tmp2_2 = (tmp1_3 - tmp1_4) / h # d2f/du2 (u=u  )
    
    
    return (tmp2_1 - tmp2_2) / h
             
def dddf_2(f,u):

    return (ddf_2(f,u + h) - ddf_2(f,u)) / h



def ddddf_1(f,u):
    
    tmp1_1_1 =      (( f(u  + 4 * h) - f(u + 3 * h)) / h )
    tmp1_1_2 =      (( f(u +  3 * h) - f(u + 2 * h)) / h )
    
    tmp1_2_1 = (tmp1_1_1 - tmp1_1_2) / h 
    
    tmp1_1_3 =      (( f(u  + 3 * h) - f(u + 2 * h)) / h )
    tmp1_1_4 =      (( f(u  + 2 * h) - f(u + 1 * h)) / h )
    
    tmp1_2_2 = (tmp1_1_3 - tmp1_1_4) / h 
    
    #######
    tmp2_1_1 = (tmp1_2_1 - tmp1_2_2) / h #  d3f/du3 (u=u+h)
    #######
    
    tmp1_1_5 =      (( f(u  + 3 * h) - f(u + 2 * h)) / h )
    tmp1_1_6 =      (( f(u  + 2 * h) - f(u + 1 * h)) / h )
    
    tmp1_2_3 = (tmp1_1_5 - tmp1_1_6) / h
    
    tmp1_1_7 =      (( f(u  + 2 * h) - f(u + 1 * h)) / h )
    tmp1_1_8 =      (( f(u  + 1 * h) - f(u + 0 * h)) / h )
    
    tmp1_2_4 = (tmp1_1_7 - tmp1_1_8) / h
    
    ########
    tmp2_2_1 = (tmp1_2_3 - tmp1_2_4) / h #  d3f/du3 (u=u  )
    ########
    
    return (tmp2_1_1 - tmp2_2_1) /h

def ddddf_2(f,u):
    
    return (dddf_2(f,u + h) - dddf_2(f,u)) / h


def dnf(f, u, n, option = "fwd"):
    
    """
        fwd
        cnt
        bkw
    """
    
    if n == 0:
        return f(u)
    else:
        if option == "fwd":
            return (dnf(f,u+h,n-1) - dnf(f,u,n-1)) /h
        elif option == "cnt":
            return (dnf(f,u+h,n-1) - dnf(f,u-h,n-1)) /h
        elif option == "bkw":
            return (dnf(f,u,n-1) - dnf(f,u-h,n-1)) /h



def dg(f,u,v):
 
    tmp1 = (g( u + 1 * h, v + 1 * h) - g( u + 0 * h, v + 1 * h)) / h #dg/du (v = v+h) 
    tmp2 = (g( u + 1 * h, v + 0 * h) - g( u + 0 * h, v + 0 * h)) / h  #dg/du (v = v) 
    

    return (tmp1 - tmp2) / h     
 

def dt(f,u,v,z):
    
    tmp1_1 = (t( u + 1 * h, v + 1 * h, z + 1 * h) - t( u + 0 * h, v + 1 * h, z + 1 * h)) / h #dg/du (v = v+h, z = z+h)
    tmp1_2 = (t( u + 1 * h, v + 0 * h, z + 1 * h) - t( u + 0 * h, v + 0 * h, z + 1 * h)) / h  #dg/du (v = v , z = z+h)
    
    tmp2_1 = (tmp1_1 - tmp1_2) / h # dg/(dudv) (z = z+h)
    
    tmp1_3 = (t( u + 1 * h, v + 1 * h, z + 0 * h) - t( u + 0 * h, v + 1 * h, z + 0 * h)) / h #dg/du (v = v+h, z = z)
    tmp1_4 = (t( u + 1 * h, v + 0 * h, z + 0 * h) - t( u + 0 * h, v + 0 * h, z + 0 * h)) / h #dg/du (v = v  , z = z)
    
    tmp2_2 = (tmp1_3 - tmp1_4) / h # dg/(dudv) (z = z)
    
    return (tmp2_1 - tmp2_2) / h
    

def d2xt(f,u,v,z):
    
    # dt/(dx2 dv)
    
    tmp1_1 = t( u + 2 * h, v + 1 * h, z) - t(u + 1 * h, v + 1 * h, z) 
    tmp1_2 = t( u + 1 * h, v + 1 * h, z) - t(u + 0 * h, v + 1 * h, z) #1
    
    tmp2_1 = (tmp1_1 - tmp1_2 )/ h #dt/d2x (v=v+h)
    
    tmp1_3 = t( u + 2 * h, v + 0 * h, z) - t(u + 1 * h, v + 0 * h, z) 
    tmp1_4 = t( u + 1 * h, v + 0 * h, z) - t(u + 0 * h, v + 0 * h, z) #1
    
    tmp2_2 = (tmp1_3 - tmp1_4)/h #dt/d2x (v=v)
    
    return (tmp2_1 - tmp2_2)/h




def dng(g, n, pos, *a):

    """
        n = (1,1,1) # derivative in the first order on all arguments of g
        *a xu,yu
    """
    
            
    if n[pos-1] <= 0 :
#         if pos >= len(n):
        return g(*a)
#         else:
#             return dng(g, n, pos + 1, *a) 
    else:
        b = list(a)
        b[pos-1] = b[pos-1] + h
        n[pos-1] = n[pos-1] - 1
        b = tuple(b)
        
        return (dng(g, n, pos, *b) - dng(g, n, pos, *a)) / h
        
        # 1 --> (f(u+h,v) - f(u,v)) /h
        # 2 --> 
        
# shock della prima su 187

    

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    
        
    xu = np.arange(1,10, 0.001)
    yu = np.arange(1,10, 0.001)
    zu = np.arange(1,10, 0.001)

    a = 2
    
    dfu = df(f,xu ) #5x**4
#     plt.plot(xu, dfu) #ok
    
    ddfu_1 = ddf_1(f, xu) # 20x**3
    ddfu_2 = ddf_2(f, xu)
#     plt.plot(xu, ddfu_1) # ok
#     plt.plot(xu, ddfu_2) # ok
    
    
    dddfu_1 = dddf_1(f,xu) #60x**2
    dddfu_2 = dddf_2(f,xu)
#     plt.plot(xu, dddfu_1) # ok
#     plt.plot(xu, dddfu_2) # ok
    
    
    ddddfu_1 = ddddf_1(f, xu) #120x
    ddddfu_2 = ddddf_2(f, xu) 
#     plt.plot(xu, ddddfu_1) # ok
#     plt.plot(xu, ddddfu_2) #ok
    
    
    dnf1 = dnf(f, xu, 1)
#     plt.plot(xu, dnf1) #ok
    
    dnf2 = dnf(f, xu, 2)
#     plt.plot(xu, dnf2) # ok
    
    dnf3 = dnf(f, xu, 3)
#     plt.plot(xu, dnf3) # ok
    
    dnf4 = dnf(f, xu, 4) 
#     plt.plot(xu, dnf4)
    
    
    
    fxua = f(xu + a)
#     plt.plot(xu, fxua, label = 'true')
    
    fyu = f(xu)
    fyu += a * dfu
#     plt.plot(xu, fyu, label = '1st')
    
    fyu += (1/spe.factorial(2) * a **2 * ddfu_1)
#     plt.plot(xu, fyu, label = '2nd' )
    
    fyu += 1/spe.factorial(3) * a**3 * dddfu_1
#     plt.plot(xu, fyu, label = '3rd')
    
    fyu += 1/spe.factorial(4) * a**3 * ddddfu_1
#     plt.plot(xu, fyu, label = '4th')
    
#     plt.legend()


    dgu = dg(g, xu, yu) 
#     plt.plot(xu, dgu)  # ok
    
    dtu = dt(g, xu, yu, zu) #2x
#     plt.plot(xu, dtu)

    dngu = dng(g, [2,0], 1, xu, yu)
    plt.plot(xu, dngu)
    
    print("")









    
    
    
    
    
     
    