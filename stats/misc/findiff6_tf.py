'''
Created on 16 Sep 2020

@author: snake91
'''


import tensorflow as tf # use venv
import numpy as np
import itertools as it


a = tf.constant(3.)
b = tf.constant(5.)

with tf.GradientTape(persistent=True) as g:
    g.watch([a,b])
    y = a * a + b
    z = y * y


dz_da = g.gradient(z, a)  # 108.0 (4*x^3 at x = 3)
dy_da = g.gradient(y, a)  # 6.0
dy_db = g.gradient(y, b)



print(dz_da)
print(dy_da)
print(dy_db)

del g  # Drop the reference to the tape


#### generate combinations to feed to tf to compute derivatives
possible_values = (1,2)
n_positions = 3

sorted_combinations = it.combinations_with_replacement(possible_values, n_positions)
unique_permutations = set()
for combo in sorted_combinations:
    # TODO: Do filtering for acceptable combinations before passing to permutations.
    for p in it.permutations(combo):
        unique_permutations.add(p)


# print "len(unique_permutations) = %i. It should be %i^%i = %i.\nPermutations:" % (len(unique_permutations), len(possible_values), n_positions, pow(len(possible_values), n_positions))
for p in unique_permutations:
    print(p)