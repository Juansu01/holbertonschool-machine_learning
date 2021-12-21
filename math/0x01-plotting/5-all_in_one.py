#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

fig = plt.figure()
fig.suptitle('All in one')
fig.add_subplot(321)
plt.plot(y0, "r-")
plt.xlim([0, 10])

fig.add_subplot(322)
plt.scatter(x1, y1, color="magenta")
plt.title('Men\'s Height vs Weight', size="x-small")
plt.xlabel('Weight (lbs)', size="x-small")
plt.ylabel('Height (in)', size="x-small")

fig.add_subplot(323)
plt.plot(x2, y2)
plt.yscale('log')
plt.title('Exponential Decay of C-14', size="x-small")
plt.ylabel('Fraction Remaining', size="x-small")
plt.xlabel('Time (years)', size="x-small")
plt.xlim([0, 28650])


fig.add_subplot(324)
l1, = plt.plot(x3, y31, '--', color="red")
l2, = plt.plot(x3, y32, color="green")
plt.xlim([0, 20000])
plt.legend((l1, l2), ('C-14', 'Ra-226'), loc='upper right', prop={'size': 4})
plt.title('Exponential Decay of Radioactive Elements', size='x-small')
plt.ylabel('Fraction Remaining', size='x-small')
plt.xlabel('Time (years)')
plt.ylim([0.0, 1.0])


fig.add_subplot(313)
bins_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
plt.hist(student_grades, bins=bins_list, edgecolor='black')
plt.xlim([0, 100])
plt.xticks(bins_list)
plt.ylim([0, 30])
plt.title("Project A", size='x-small')
plt.xlabel("Grades", size='x-small')
plt.ylabel("Number of Students", size='x-small')


fig.tight_layout()
plt.show()
