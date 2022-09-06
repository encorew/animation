import random
import time

import numpy as np
import matplotlib.pyplot as plt

a = []
b = []
c = []
for i in range(50):
    a.append(random.uniform(0, 1))
for i in range(50):
    b.append(a[i] + random.uniform(-0.2, 0.2))
for i in range(50):
    c.append(a[i] + random.uniform(-0.2, 0.2))
plt.subplot(3,1,1)
plt.plot(a)
plt.subplot(3,1,2)
plt.plot(b,color='red')
plt.subplot(3,1,3)
plt.plot(c,color='black')
plt.show()
