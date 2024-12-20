'''
Created on Dec 8, 2018

@author: snake91
'''


from stats.mle import simulate as sim
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import confidence_interval as ci

window = 250
phi = -0.9
sigma = 1
# u = int(np.random.randint())
u = 15

np.random.seed(u)
x = np.random.normal(size = 500)

np.random.seed(u)
y = sim.ar1Gaussian(t = 500, phi = phi, y0 = 0.5)


var_x = [np.percentile(x[i-window : i], q = 0.99) for i in range(window, 500)]
var_y_iid = [np.percentile(y[i-window : i], q = 0.99) for i in range(window, 500)]
var_y_ar = ci.arpGaussian(y, phi = [phi], window = 250, q = 0.99, sigma = 1)[1]

# plt.plot(x[window:], label = 'iid')
plt.plot(y[window:], label = 'ar1')
# plt.plot(var_x, label = 'var_iid_iid')
plt.plot(var_y_iid, label = 'var_ar1_iid')
plt.plot(var_y_ar, label = 'var_ar1_ar1')
plt.legend()

breach_y_iid = filter(lambda x: x==True, [ y[i] < var_y_iid[i-window] for i in range(window, 500)])
breach_y_ar1 = filter(lambda x: x==True, [ y[i] < var_y_ar[i-window] for i in range(window, 500)])

print('iid', len(breach_y_iid))
print('ar1', len(breach_y_ar1))