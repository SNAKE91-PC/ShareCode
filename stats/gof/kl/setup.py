'''
Created on 26 Dec 2019

@author: snake91
'''

from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize

import numpy as np
import sys


path = "/home/snake91/git/ShareCode/stats/gof/kl"

sys.path.append(path)
# extensions = [Extension('kullback', ['kullback.pyx']                        
#                         )
#               ]

setup(ext_modules = cythonize("/home/snake91/git/ShareCode/stats/gof/kl/kullback.pyx", 
                                language_level = sys.version_info[0], 
                                language = "c++"),
                               
        include_dirs=[np.get_include()]
        )