
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from copy import deepcopy
# from scipy.stats import gaussian_kde



        
        
class AnimatedScatter(object):
    """An animated scatter plot using matplotlib.animations.FuncAnimation."""
    def __init__(self, datagen, nsmooth = 60, fig = None, ax = None, cmap = None, frames = None):
#         self.numpoints = numpoints
        self.stream = self.data_stream(datagen)

        self.nsmooth = nsmooth

        self.minxaxis = -4
        self.minyaxis = -4
        
        self.maxxaxis = 4
        self.maxyaxis = 4
        # Setup the figure and axes...
        
        if fig == None:
            self.fig, self.ax = plt.subplots(dpi = 150)
            self.cmap = cmap
        else:    
            self.fig = fig
            self.ax = ax
            self.cmap = cmap
            
        # Then setup FuncAnimation.
        if frames == None:
            frames = 660
            
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=1, frames = frames, repeat = False,
                                          init_func=self.setup_plot, blit=False)

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        x, y, z = next(self.stream).T #, s, c #, z 
#         print(x)
#         print(y)
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        self.ax.axis([-4, 4, -4, 4])

        self.scat = self.ax.scatter(x, y, c= z, s=1, cmap = self.cmap)
        plt.colorbar(self.scat)
        
        
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def smooth(self, args, n):
        
        for t in np.linspace(0, 1 + 1/n, n):
            
            interpolation = args[0]*(1-t) + args[1]*t

            yield interpolation


    def data_stream(self, datagen):

#         z = gaussian_kde(xy[:,0])(xy[:,1])
#         s, c = np.ones((self.numpoints, 2)).T/100
#         s = np.ones((self.numpoints))

        i = 0
        while True:
            
            if i == 0:
                Arr2Smooth = []
                for i in range(0, 2): # @Unusedvariable
                    Arr2Smooth.append(next(datagen))
            else:
                Arr2Smooth = Arr2Smooth[-1:]
                Arr2Smooth.append(next(datagen))
    
            smoothedData = self.smooth(Arr2Smooth, self.nsmooth)
                    
            for arr in smoothedData:
#             for arr in smoothedData:
#                   
#                 z = st.multivariate_normal.pdf(arr, mean = [0,0], cov = [[1,0],[0,1]])
#                     
#                 idx = z.argsort()
#                 x, y, z = arr[:,0][idx], arr[:,1][idx], z[idx]
#     #             xy = np.random.normal(size = (self.numpoints, 2), scale = 1) 
#     #             z = st.multivariate_normal.cdf(xy, mean = [0,0], cov = [[1,0],[0,1]])
#     #             z = gaussian_kde(xy[:,0])(xy[:,1])
#     #             s += 0.05 * (np.random.random(self.numpoints) - 0.5)
#     #             c += 0.02 * (np.random.random(self.numpoints) - 0.5)
#     #             yield np.c_[xy[:,0], xy[:,1], z]#, s, c]
#     
#                 yield np.c_[x, y, z]

                yield arr
            
            i+= 1
            

    def update(self, i):
        """Update the scatter plot."""
        
        data = next(self.stream)

        print(i)
        # Set x and y data...
        
        self.scat.set_offsets(data[:, :2])
        
        self.scat.set_label("$t = "+ str(int(i/self.nsmooth)) + "$")
        plt.legend(loc = "upper left")
        
#         if i %  30:
#             self.scat.axes.set_xlim(xmin = np.min(data[:, 0]), xmax = np.max(data[:, 0]))
#             self.scat.axes.set_ylim(ymin = np.min(data[:, 1]), ymax = np.max(data[:, 1]))



        # Set sizes...
#         self.scat.set_sizes(300 * abs(data[:, 2])**1.5 + 100)
        # Set colors..
#         self.scat.set_array(data[:, 3])

            
        if self.maxxaxis < np.max(data[:, 0]):
            self.maxxaxis = np.max(data[:, 0])  

        if self.maxyaxis < np.max(data[:, 1]):
            self.maxyaxis = np.max(data[:, 1])
            
        if self.minxaxis > np.min(data[:, 0]):        
            self.minxaxis = np.min(data[:, 0])
            
        if self.minyaxis > np.min(data[:, 1]):
            self.minyaxis = np.min(data[:, 1])
    
    
        self.ax.axis([self.minxaxis, self.maxxaxis, self.minyaxis, self.maxyaxis]) 
        
        
        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,

    def save(self, path):
        
        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=self.nsmooth, metadata=dict(artist='Snake91'), bitrate = 1000) #, bitrate=8000, codec = "h264"
        
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






    
    
    
    