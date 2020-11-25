import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import uniform, sample
from math import pi, cos, sin, sqrt
from numpy.random import rand

# Circle Parameters
radius = 1
point = (2, 2)

# Simulation Parameters
num_particles = 100
num_steps = 100
interval = 1000

# Plot Setup
figure = plt.figure()

p1 = figure.add_subplot(221)
xdata1, ydata1 = np.zeros(num_particles), np.zeros(num_particles)
ln1, = p1.plot([], [], 'bo')

p2 = figure.add_subplot(222)
xdata2, ydata2 = [], []
ln2, = p2.plot([], [], '-go')


def init_p1():
    p1.set_xlabel('X')
    p1.set_ylabel('Y')
    p1.set_xlim(-10, 10)
    p1.set_ylim(-10, 10)
    p1.set_aspect(1)
    circle = plt.Circle(point, radius, fc='r')
    p1.add_patch(circle)
    return ln1,


def init_p2():
    p2.set_xlabel('Steps')
    p2.set_ylabel('Particles in Area')
    p2.set_xlim(0, num_steps)
    p2.set_ylim(0, 20)
    return ln2,


def update_p1(frame):
    for particle in range(num_particles):
        phi = uniform(-pi, pi)
        dx = cos(phi)
        dy = sin(phi)
        xdata1[particle] += dx
        ydata1[particle] += dy
    ln1.set_data(xdata1, ydata1)
    return ln1,


def update_p2(frame):
    n = len([p for p in range(num_particles) if
             (sqrt((xdata1[p] - point[0]) ** 2 + (ydata1[p] - point[1]) ** 2) <= radius)])
    xdata2.append(frame)
    ydata2.append(n)
    ln2.set_data(xdata2, ydata2)
    return ln2,


ani1 = FuncAnimation(figure, update_p1, frames=num_steps,
                     init_func=init_p1, interval=interval, blit=True, repeat=False)

ani2 = FuncAnimation(figure, update_p2, frames=num_steps,
                     init_func=init_p2, interval=interval, blit=True, repeat=False)

p1.grid()
p2.grid()
plt.show()
