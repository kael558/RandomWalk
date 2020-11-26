import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import uniform, sample
from math import pi, cos, sin, sqrt
from numpy.random import rand
import random

'''
DONE
line plot of mean distance from origin - d
line plot of mean distance squared from origin -d

TODO
3-d plot?
sample steps from mean free path = 1/(sqrt(2)*number density*size of molecules) instead of 1
requires:
include bounce of walls (if greater than bounds, simulate bounce back) -d

include discussion on diffusion, effusion
'''

# Circle Parameters
radius = 1
point = (2, 2)

# Simulation Parameters
num_particles = 10
num_steps = 100
interval = 1000
box_size = 10  # square with side lengths = box_size*2

molecule_size = 10
number_density = num_particles / ((box_size * 2) ** 2)


# Plot Setup
figure = plt.figure()

p1 = figure.add_subplot(221)
ln1, = p1.plot([], [])
colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
          for i in range(num_particles)]

lines = []
for i in range(num_particles):
    lobj = p1.plot([0], [0], color=colors[i], markevery=[-1], marker='o', linewidth=1)[0]
    lines.append(lobj)

# Holding values for efficiency to be used in the next 3 sub plots
step_count = []
distances = np.zeros(num_particles)

p2 = figure.add_subplot(222)
ln2, = p2.plot([], [], color='red', marker='.', linestyle='solid')

p3 = figure.add_subplot(223)
ln3, = p3.plot([], [], color='green', marker='.', linestyle='solid')

p4 = figure.add_subplot(224)
ln4, = p4.plot([], [], color='blue', marker='.', linestyle='solid')


def init_p1():
    p1.set_xlabel('X')
    p1.set_ylabel('Y')
    p1.set_xlim(-box_size, box_size)
    p1.set_ylim(-box_size, box_size)

    circle = plt.Circle(point, radius, fc='r')
    p1.add_patch(circle)
    return lines


def init_p2():
    p2.set_xlabel('Steps')
    p2.set_ylabel('Particles in Area')
    p2.set_xlim(0, num_steps)
    p2.set_ylim(0, num_particles)
    return ln2,


def init_p3():
    p3.set_xlabel('Steps')
    p3.set_ylabel('Average Distance')
    p3.set_xlim(0, num_steps)
    p3.set_ylim(0, sqrt(num_steps))
    return ln3,


def init_p4():
    p4.set_xlabel('Steps')
    p4.set_ylabel('Average Distance Squared')
    p4.set_xlim(0, num_steps)
    p4.set_ylim(0, num_steps)
    return ln4,


def update_p1(frame):
    for p, line in enumerate(lines):
        phi = uniform(-pi, pi)
        dx = cos(phi)
        dy = sin(phi)

        temp_x = line.get_xdata()[-1] + dx
        temp_y = line.get_ydata()[-1] + dy

        if abs(temp_x) >= box_size:
            temp_x = 2 * (box_size if temp_x > 0 else -box_size) - temp_x

        if abs(temp_y) >= box_size:
            temp_y = 2 * (box_size if temp_y > 0 else -box_size) - temp_y

        list_x = np.append(line.get_xdata(), temp_x)
        list_y = np.append(line.get_ydata(), temp_y)

        line.set_data(list_x, list_y)
        distances[p] = sqrt((temp_x - point[0]) ** 2 + (temp_y - point[1]) ** 2)
    step_count.append(frame)

    return lines


def update_p2(frame):
    n = len([p for p in range(num_particles) if (distances[p] <= radius)])
    ydata2 = np.append(p2.lines[0].get_ydata(), n)
    ln2.set_data(step_count, ydata2)
    return ln2,


def update_p3(frame):
    avg = sum(distances) / num_particles
    ydata3 = np.append(p3.lines[0].get_ydata(), avg)
    ln3.set_data(step_count, ydata3)
    return ln3,


def update_p4(frame):
    avg = sum(distances) / num_particles
    ydata4 = np.append(p4.lines[0].get_ydata(), avg ** 2)
    ln4.set_data(step_count, ydata4)
    return ln4,


ani1 = FuncAnimation(figure, update_p1, frames=num_steps,
                     init_func=init_p1, interval=interval, blit=True, repeat=False)
p1.grid()

ani2 = FuncAnimation(figure, update_p2, frames=num_steps,
                     init_func=init_p2, interval=interval, blit=True, repeat=False)
p2.grid()
ani3 = FuncAnimation(figure, update_p3, frames=num_steps,
                     init_func=init_p3, interval=interval, blit=True, repeat=False)
p3.grid()
ani4 = FuncAnimation(figure, update_p4, frames=num_steps,
                     init_func=init_p4, interval=interval, blit=True, repeat=False)
p4.grid()

plt.show()
