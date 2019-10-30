'''
Created on Dec 8, 2018

@author: snake91
'''


from mle import simulate as sim
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import prediction.confidence_interval as ci

window = 250
phi = [-0.6, -0.3]
sigma = 1
u = int(np.random.randint(1000))
# u = 15

np.random.seed(u)
x = np.random.normal(size = 500)

np.random.seed(u)
y = sim.arpGaussian(t = 500, phi = phi, y0 = list(np.random.normal(size = len(phi))))


var_x = [np.percentile(x[i-window : i], q = 0.99) for i in range(window, 500)]
var_y_iid = [np.percentile(y[i-window : i], q = 0.99) for i in range(window, 500)]

var_y_ar = ci.arpGaussian(y, phi = phi, window = 250, q = 0.99, sigma = 1)[1]


# plt.plot(x[window:], label = 'iid')
plt.plot(y[window:], label = 'arp')
# plt.plot(var_x, label = 'var_iid_iid')
plt.plot(var_y_iid, label = 'var_arp_iid')
plt.plot(var_y_ar, label = 'var_arp_arp')
plt.legend()

breach_y_iid = filter(lambda x: x==True, [ y[i] < var_y_iid[i-window] for i in range(window, 500)])
breach_y_arp = filter(lambda x: x==True, [ y[i] < var_y_ar[i-window] for i in range(window, 500)])

print 'iid', len(breach_y_iid)
print 'arp', len(breach_y_arp)


print("")