#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

fig, ax = plt.subplots()
l1, = plt.plot(x, y1, '--', color="red")
l2, = plt.plot(x, y2, color="green")
plt.xlim([0, 20000])
plt.ylim([0, 1])
ax.legend((l1, l2), ('C-14', 'Ra-226'), loc='upper right')
ax.set_title('Exponential Decay of Radioactive Elements')
ax.set_ylabel('Fraction Remaining')
ax.set_xlabel('Time (years)')
plt.show()
