'''
Created on 23 May 2020

@author: snake91
'''

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt



class Brownian2d:
    
    def __init__(self, people):

        self.people = people
        self.particles_init = np.random.uniform(size = (self.people,2))
        self.scale = 0.8 # could be a vector
        
    def simulate(self, cov):
        
        particles = self.particles_init
        
        f = lambda x: st.multivariate_normal.cdf(x, mean = np.zeros(shape = len(x)), cov = cov)
        
        
        while True:
                
            rnd = self.cholesky(np.random.normal(size = (self.people, 2), loc = 0, scale = 1), cov) #not correlated
            particles = particles + rnd * (self.scale**2)
            
            cdf = np.array(list(map(lambda x: f(x), rnd)))  
            
            yield np.c_[particles[:,0], particles[:,1], cdf]

    
    def cholesky(self, rnd, cov):
        
        arr = np.array(list(map(lambda x: np.dot(np.linalg.cholesky(cov), x), rnd)))
        
        return arr
        
        
        
if __name__ == '__main__':
     
     
    task = "scatter"
     
    from cpxnetw.animplot.animscatfunc import AnimatedScatter
    from cpxnetw.animplot.animlinefunc import AnimatedLine

    
    
    task = "scatter"
    
    if task == "scatter":
        engine1 = Brownian2d(people = 1000)
        engine2 = Brownian2d(people = 1000)
        
        cov1 = [[1,-0.9],[-0.9,1]]
        particles1 = engine1.simulate(cov1)
        
        cov2 = [[1,0.9],[0.9,1]]
        particles2 = engine2.simulate(cov2)
        
        scat1 = AnimatedScatter(particles1, nsmooth = 120, cmap = 'Blues', frames = 100)
        scat2 = AnimatedScatter(particles2, nsmooth = 120, fig = scat1.fig, ax = scat1.ax, cmap = "Reds", frames = 100)
        
#         plt.show()
        
        scat1.save(r"D:\1.mp4")
        
    if task == "line":
        
        cov = [[1,0.0],[0.0,1]]
        
        engine1 = Brownian2d(people = 1)
        engine2 = Brownian2d(people = 1)
        engine3 = Brownian2d(people = 1)

        particles1 = engine1.simulate(cov)
        particles2 = engine2.simulate(cov)
        particles3 = engine3.simulate(cov)
    #     scat = AnimatedScatter(particles)
        line1 = AnimatedLine(particles1, nsmooth = 1)
        line2 = AnimatedLine(particles2, nsmooth = 1, fig = line1.fig, ax = line1.ax)
        line3 = AnimatedLine(particles3, nsmooth = 1, fig = line1.fig, ax = line1.ax)
        
        plt.show()
    
#     print("")
     
#     scat.save("/home/snake91/animation.mp4")
#     line.save("/home/snake91/line_animation.mp4")
    
    
    