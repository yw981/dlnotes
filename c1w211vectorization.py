import numpy as np
import time

NUM = 100000
a = np.random.rand(NUM)
b = np.random.rand(NUM)

tic = time.time()
c = np.dot(a, b)
toc = time.time()
print('vectorized version : %f ms and c = %f ' % (1000*(toc-tic), c))

tic = time.time()
for i in range(NUM):
    c += a[i] * b[i]

toc = time.time()
print('For loop : %f ms and c = %f ' % (1000*(toc-tic), c))

# Informally: There are functions you can compute with a
# “small” L-layer deep neural network that shallower networks
# require exponentially more hidden units to compute.
