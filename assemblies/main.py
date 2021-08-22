


import clr
import numpy as np

import sys
import matplotlib.pyplot as plt


sys.path.append(r"C:\Users\Snake91\git\ShareCode\assemblies\Assemblies\bin\Debug")


clr.AddReference(r"C:\Users\Snake91\git\ShareCode\assemblies\Assemblies\bin\Debug\net40\Assemblies.dll")

import Assemblies 
 


# print(Assemblies.BlackScholes)


r = np.arange(0.01,0.2, 0.001)
S0 = np.array([100] * len(r))
sigma = np.array([0.5] * len(r))
T = np.array([10/365.] * len(r))

x = Assemblies.BlackScholes().simulatePrice(r, S0, sigma, T)

list(x)

print("")