'''
Created on 2 Feb 2020

@author: snake91
'''


import numpy as np


def copulamle(theta, f, args):
    
    C = args[-1]
    args = args[:-1]
    
    x = f(*args, theta)

    try:       
        res = np.log(np.mean((x - C)**2))
    except RuntimeWarning:
        res = -10e12
#     print(theta[0], res, sep = " ")
    return res
