
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy.stats as st
# from scipy.stats import gaussian_kde


class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, people=10000):
        self.people = people
        self.stream = self.data_stream()

        # Setup the figure and axes...
        self.fig, self.ax = plt.subplots(dpi = 200)
        # Then setup FuncAnimation.
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=0.001, frames = 300, repeat = False,
                                          init_func=self.setup_plot, blit=False)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y, z = next(self.stream).T #, s, c
#         print(x)
#         print(y)
        self.scat = self.ax.scatter(x, y, c= z, s=1)
        self.ax.axis([-4, 4, -4, 4])
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def smooth(self, args, n):

        for t in np.linspace(0, 1 + 1/n, n):
            
            interpolation = args[0]*(1-t) + args[1]*t

            yield interpolation


    def data_stream(self):

#         z = gaussian_kde(xy[:,0])(xy[:,1])
#         s, c = np.ones((self.people, 2)).T/100
#         s = np.ones((self.people))
        while True:
            Arr2Smooth = []
            for i in range(0, 2): # @Unusedvariable
                
                x = np.random.normal(size = self.people)
                y = np.random.normal(size = self.people)
                xy = np.c_[x, y]
                
                Arr2Smooth.append(xy)
    
            smoothedData = self.smooth(Arr2Smooth, 120)
                    
            #for arr in next(smoothedData):
            for arr in smoothedData:
                  
                z = st.multivariate_normal.pdf(arr, mean = [0,0], cov = [[1,0],[0,1]])
                    
                idx = z.argsort()
                x, y, z = arr[:,0][idx], arr[:,1][idx], z[idx]
    #             xy = np.random.normal(size = (self.people, 2), scale = 1) 
    #             z = st.multivariate_normal.cdf(xy, mean = [0,0], cov = [[1,0],[0,1]])
    #             z = gaussian_kde(xy[:,0])(xy[:,1])
    #             s += 0.05 * (np.random.random(self.people) - 0.5)
    #             c += 0.02 * (np.random.random(self.people) - 0.5)
    #             yield np.c_[xy[:,0], xy[:,1], z]#, s, c]
    
                yield np.c_[x, y, z]


    def update(self, i):
        """Update the scatter plot."""
        
        data = next(self.stream)

        print(i)
        # Set x and y data...
        
        self.scat.set_offsets(data[:, :2])
        
        
#         if i %  30:
#             self.scat.axes.set_xlim(xmin = np.min(data[:, 0]), xmax = np.max(data[:, 0]))
#             self.scat.axes.set_ylim(ymin = np.min(data[:, 1]), ymax = np.max(data[:, 1]))



        # Set sizes...
#         self.scat.set_sizes(300 * abs(data[:, 2])**1.5 + 100)
        # Set colors..
#         self.scat.set_array(data[:, 3])
        
#         self.ax.axis([np.min(data[:, 0]), np.max(data[:, 0]), np.min(data[:, 1]), np.max(data[:, 1])]) 
        
        
        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def save(self, path):
        
        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=5000)
        
        self.ani.save(path, writer=writer)
    

# def _blit_draw(self, artists, bg_cache):
#     # Handles blitted drawing, which renders only the artists given instead
#     # of the entire figure.
#     updated_ax = []
#     for a in artists:
#         # If we haven't cached the background for this axes object, do
#         # so now. This might not always be reliable, but it's an attempt
#         # to automate the process.
#         if a.axes not in bg_cache:
#             # bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.bbox)
#             # change here
#             bg_cache[a.axes] = a.figure.canvas.copy_from_bbox(a.axes.figure.bbox)
#         a.axes.draw_artist(a)
#         updated_ax.append(a.axes)
# 
#     # After rendering all the needed artists, blit each axes individually.
#     for ax in set(updated_ax):
#         # and here
#         # ax.figure.canvas.blit(ax.bbox)
#         ax.figure.canvas.blit(ax.figure.bbox)





if __name__ == '__main__':
    
    
#     matplotlib.animation.Animation._blit_draw = _blit_draw
    a = AnimatedScatter()
#     plt.show()
    
    a.save("/home/snake91/animation.mp4")
    
    
    
    
    