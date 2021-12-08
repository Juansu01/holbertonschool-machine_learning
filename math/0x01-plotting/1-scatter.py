#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
fig, ax = plt.subplots()

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x, y = np.random.multivariate_normal(mean, cov, 2000).T
y += 180

plt.scatter(x, y, color="magenta")
ax.set_title('Men\'s Height vs Weight')
ax.set_ylabel('Weight (lbs)')
ax.set_xlabel('Height (in)')
plt.show()
