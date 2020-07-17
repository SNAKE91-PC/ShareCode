'''
Created on 4 Jul 2020

@author: snake91
'''


import scipy.fft as fft
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image
from matplotlib import cm
import os
import numpy as np

path = os.path.dirname(__file__) + "/London-BW.jpg" 

pic = Image.open(path).convert('L')

# pic = io.imread(path, as_gray=True)
pix = np.asarray(pic) 

plt.imshow(pix, cmap=cm.gray, vmin=0, vmax=255)

pixn = pix + np.random.normal(loc = 0, scale = 40, size = pix.shape)
pixn = pixn.astype(int)
plt.figure()
plt.imshow(pixn, cmap=cm.gray, vmin=0, vmax=255)

pixdenoise = pixn - np.random.normal(loc = 0, scale = 40, size = pix.shape)
pixdenoise = pixdenoise.astype(int)
plt.figure()
plt.imshow(pixdenoise, cmap=cm.gray, vmin=0, vmax=255)


pif = fft.fft2(pixn)
plt.imshow(np.abs(pif), cmap=cm.gray, vmin=0, vmax=255)
plt.colorbar()
 
pifn = pif - np.random.normal(loc = 0, scale = 40, size = pix.shape)
pixm = fft.ifft2(pifn)
pixm = pixm.astype(int)
plt.figure()
plt.imshow(pixm, cmap=cm.gray, vmin=0, vmax=255)

