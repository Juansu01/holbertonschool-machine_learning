#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

labels = ['Farrah', 'Fred', 'Felicia']
width = 0.5

fig, ax = plt.subplots()

ax.bar(labels, fruit[0], width, label='apples', color='red')
ax.bar(labels, fruit[1], width, bottom=fruit[0], color='yellow',
       label='bananas')
ax.bar(labels, fruit[2], width, bottom=fruit[0] + fruit[1], label='oranges',
       color='#ff8000')
ax.bar(labels, fruit[3], width, bottom=fruit[0] + fruit[1] + fruit[2],
       color='#ffe5b4', label='peaches')

ax.set_ylabel('Quantity of Fruit')
ax.set_title('Number of Fruit per Person')
ax.legend()
plt.ylim([0, 80])

plt.show()
