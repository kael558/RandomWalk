import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import uniform
from math import pi, cos, sin, sqrt

# Circle Parameters
radius = 1
point = (5, 5)

# Simulation Parameters
num_particles = 100
num_steps = 100
interval = 1000

# Plot Setup
figure = plt.figure()

p1 = figure.add_subplot(211)
xdata1, ydata1 = [0] * num_particles, [0] * num_particles
ln1, = plt.plot([], [], 'bo')

p2 = figure.add_subplot(212)
xdata2, ydata2 = [], []
ln2, = p2.plot([], [], 'go')


def init_p1():
    p1.set_xlabel('X')
    p1.set_ylabel('Y')
    p1.set_xlim(-10, 10)
    p1.set_ylim(-10, 10)
    return ln1,


def init_p2():
    p2.set_xlabel('Steps')
    p2.set_ylabel('Particles in Area')
    p2.set_xlim(0, num_steps)
    p2.set_ylim(0, num_particles)
    return ln2,


def update_p1(frame):
    for p in range(num_particles):
        phi = uniform(-pi, pi)
        dx = cos(phi)
        dy = sin(phi)
        xdata1[p] += dx
        ydata1[p] += dy
    # plt.Circle((circle_point[0], circle_point[1]), circle_radius)
    ln1.set_data(xdata1, ydata1)
    return ln1,


def update_p2(frame):
    n = 0

    for p in range(num_particles):
        d = sqrt((xdata1[p] - point[0]) ** 2 + (ydata1[p] - point[1]) ** 2)
        if d <= radius:
            n += 1

    xdata2.append(frame)
    ydata2.append(n)
    ln2.set_data(xdata2, ydata2)
    return ln2,


ani1 = FuncAnimation(figure, update_p1, frames=np.linspace(0, num_steps, num_steps + 1),
                     init_func=init_p1, interval=interval, blit=True, repeat=False)

ani2 = FuncAnimation(figure, update_p2, frames=np.linspace(0, num_steps, num_steps + 1),
                     init_func=init_p2, interval=interval, blit=True, repeat=False)

p1.grid()
p2.grid()
plt.show()
