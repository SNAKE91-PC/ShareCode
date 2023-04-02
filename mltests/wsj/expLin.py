

import numpy as np
import scipy.optimize as spo
import math
import matplotlib.pyplot as plt

def fun(a,x):

    res = (np.log(x) / np.log(a)) - x

    # res =  math.log(0.01, a) - 0.01

    # print(a,x,res)
    return res

# res = spo.bisect(fun, 10e-6, 10, args=(0.5,))
#
# print("")


a = np.array(list(range(10,10**5, 10**3)))
y = fun(a,10**5)

plt.figure()
plt.plot(a, y)

plt.show()

print("")


