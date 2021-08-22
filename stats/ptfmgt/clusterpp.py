'''
Created on 30 Aug 2020

@author: snake91
'''


def host(idhost):
    import socket
    from math import sin
    sumhost=0
    for i in range(1,1000):
        for j in range(1,1000):
            sumhost=sumhost+sin(i/j)
    
    return "Rank: %d -- %s. %f" % (idhost, socket.gethostname(),sumhost+idhost)



if __name__ == "__main__":
    
    from pathos.parallel import ParallelPool as Pool # @Unresolvedimport
#     import pathos.pools as pp
    import time
    
    pool = Pool()
#     pool = pp.ProcessPool()

    pool.ncpus = 20
    print("Evaluate on "+str(pool.ncpus)+" cpus")
    
    pool.servers = ('192.168.1.109:5654',) #'192.168.1.103:5653', 
    
    res5 = pool.amap( host, range(25) )
    while not res5.ready():
        time.sleep(1)
    
    
    print(  "\n".join(res5.get()) )