'''
Created on Jan 12, 2019

@author: snake91
'''


from mle.mleclass import mleobj
from mle import simulate as sim

x = sim.arpGaussian(t = 500, phi = [0.5, 0.2])

obj = mleobj(x, order = (2,0))

res = obj.fit()

print(res)


