#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

bins_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
fig, ax = plt.subplots()
plt.hist(student_grades, bins=bins_list, edgecolor='black')
plt.xlim([0, 100])
plt.xticks(bins_list)
plt.ylim([0, 30])
plt.title("Project A")
plt.xlabel("Grades")
plt.ylabel("Number of Students")
plt.show()
