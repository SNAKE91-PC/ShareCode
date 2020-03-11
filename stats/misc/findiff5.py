'''
Created on 17 Feb 2020

@author: snake91
'''

import numpy as np
import scipy.optimize as spo
# import numdifftools as nd
# 
# xdata = np.reshape(np.arange(0,1,0.1),(-1,1))
# ydata = 1+2*np.exp(0.75*xdata)
# 
# 
# def fun(c):
#     return (c[0]+c[1]*np.exp(c[2]*xdata) - ydata)**2
# 
# 
# Jfun = nd.Jacobian(fun)
# np.allclose(np.abs(Jfun([1,2,0.75])), 0) # should be numerically zero


# print("")




import algopy
from algopy import UTPM

import matplotlib.pyplot as plt

def f(x):
    
    return np.sin(x[0]**2 + 5)


def gradient(x):
    
    out = []
    if type(x) == float:
        i = UTPM.init_jacobian([x])
        y = f(i)
        algopy_jacobian = UTPM.extract_jacobian(y)
        out = algopy_jacobian
    else:
        for i in x:
            i = UTPM.init_jacobian([i])
            y = f(i)
            algopy_jacobian = UTPM.extract_jacobian(y)
        #     print('jacobian = ',algopy_jacobian)
            out.append(algopy_jacobian[0])
    
    return np.array(out)
    



def eval_f(x):
    """ some function """
    return x[0]**2 * x[1]**2#x[1]*x[2] + np.exp(x[0])*x[1]

# forward mode without building the computational graph
# -----------------------------------------------------
x = UTPM.init_jacobian([10,10])
y = eval_f(x)
algopy_jacobian = UTPM.extract_jacobian(y)


# xax = np.linspace(-8,8,100)
# gradient(xax)

# right
a = np.linspace(1,10,1000)

# plt.plot(a, f([a]))
# xopt = spo.brenth(lambda x: gradient(x), 0, 20, xtol = 10e-7, full_output = True)


h = gradient(a)
plt.plot(a, h)


plt.plot(a,f([a]))
# print('jacobian = ',algopy_jacobian)

# reverse mode using a computational graph
# ----------------------------------------

# STEP 1: trace the function evaluation
cg = algopy.CGraph()
x = algopy.Function([1,2])
y = eval_f(x)
cg.trace_off()
cg.independentFunctionList = [x]
cg.dependentFunctionList = [y]

# STEP 2: use the computational graph to evaluate derivatives
print('gradient =', cg.gradient([10.,10]))
print('Jacobian =', cg.jacobian([10.,10]))
print('Hessian =', cg.hessian([10.,10.]))
print('Hessian vector product =', cg.hess_vec([3.,5.],[4,5]))




print("")















