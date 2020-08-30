'''
Created on 30 Aug 2020

@author: snake91
'''

from pathos.parallel import ParallelPool as Pool
import time
pool = Pool()

def host(idhost):
    import socket
    from math import sin
    sumhost=0
    for i in range(1,1000):
        for j in range(1,10000):
            sumhost=sumhost+sin(i/j)
    
    return "Rank: %d -- %s. %f" % (idhost, socket.gethostname(),sumhost+idhost)


pool.ncpus = 20
print("Evaluate on "+str(pool.ncpus)+" cpus")

pool.servers = ('192.168.1.113:5653', '192.168.1.109:5654')

res5 = pool.amap( host, range(16) )
while not res5.ready():
    time.sleep(1)


print(  "\n".join(res5.get()) )