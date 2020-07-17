'''
Created on 23 May 2020

@author: snake91
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vispy: gallery 60

"""
Dynamic planar graph layout.
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vispy: gallery 2
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

"""
Multiple real-time digital signals with GLSL-based clipping.
"""

from vispy import gloo
from vispy import app
import numpy as np
import math

# Number of cols and rows in the table.
nrows = 1
ncols = 1

# Number of signals.
m = nrows*ncols

# Number of samples per signal.
n = 1000

# Various signal amplitudes.
amplitudes = .1 + .2 * np.random.rand(m, 1).astype(np.float32)

# Generate the signals as a (m, n) array.
y = amplitudes * np.random.randn(m, n).astype(np.float32)

# Color of each vertex (TODO: make it more efficient by using a GLSL-based
# color map and the index).
color = np.repeat(np.random.uniform(size=(m, 3), low=.5, high=.9),
                  n, axis=0).astype(np.float32)

# Signal 2D index of each vertex (row and col) and x-index (sample index
# within each signal).
index = np.c_[np.repeat(np.repeat(np.arange(ncols), nrows), n),
              np.repeat(np.tile(np.arange(nrows), ncols), n),
              np.tile(np.arange(n), m)].astype(np.float32)

VERT_SHADER = open("/home/snake91/git/ShareCode/stats/cpxnetw/vispyplot/vertexshader.cpp").read() 
FRAG_SHADER = open("/home/snake91/git/ShareCode/stats/cpxnetw/vispyplot/fragshader.cpp").read()


class Canvas(app.Canvas):
    def __init__(self):
        app.Canvas.__init__(self, title='Use your wheel to zoom!', size = (1280,720),
                            keys='interactive')
        self.program = gloo.Program(VERT_SHADER, FRAG_SHADER)
        self.program['a_position'] = y.reshape(-1, 1)
        self.program['a_color'] = color
        self.program['a_index'] = index
        self.program['u_scale'] = (1., 1.)
        self.program['u_size'] = (nrows, ncols)
        self.program['u_n'] = n

        gloo.set_viewport(0, 0, *self.physical_size)

        self._timer = app.Timer('auto', connect=self.on_timer, start=True)

        gloo.set_state(clear_color='black', blend=True,
                       blend_func=('src_alpha', 'one_minus_src_alpha'))

        self.show()

    def on_resize(self, event):
        gloo.set_viewport(0, 0, *event.physical_size)

    def on_mouse_wheel(self, event):
        dx = np.sign(event.delta[1]) * .05
        scale_x, scale_y = self.program['u_scale']
        scale_x_new, scale_y_new = (scale_x * math.exp(2.5*dx),
                                    scale_y * math.exp(0.0*dx))
        self.program['u_scale'] = (max(1, scale_x_new), max(1, scale_y_new))
        self.update()

    def on_timer(self, event):
        """Add some data at the end of each signal (real-time signals)."""
        k = 100
        y[:, :-k] = y[:, k:] # here it deletes part of the numbers
        y[:, -k:] = amplitudes * np.random.randn(m, k) # here it adds more numbers

        self.program['a_position'].set_data(y.ravel().astype(np.float32))
        self.update()

    def on_draw(self, event):
        gloo.clear()
        self.program.draw('line_strip')

if __name__ == '__main__':
    c = Canvas()
    app.run()
    
    
    
    