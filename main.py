import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import uniform
from math import pi, cos, sin

figure = plt.figure()
axes = plt.axes()
# draw_circle = plt.Circle((0.5, 0.5), 0.3)
# axes.set_aspect(1)
# axes.add_artist(draw_circle)

num_particles = 100
num_steps = 100

circle_point = (5, 5)
circle_radius = 1

xdata, ydata = [0] * num_particles, [0] * num_particles

# p2 = fig.add_subplot(222)

ln1, = plt.plot([], [], 'bo')

# ln2, = p2.plot([], [], 'go')


def init_p1():
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_xlim(-10, 10)
    axes.set_ylim(-10, 10)
    return ln1,


'''
def initP2():
    ax = p2.axes()
    ax.set_xlabel('Steps')
    ax.set_ylabel('Particles in Area')
    ax.set_xlim(0, num_steps)
    ax.set_ylim(0, num_particles)
    return ln2,
'''


def update_p1(frame):
    for particle in range(num_particles):
        phi = uniform(-pi, pi)
        dx = cos(phi)
        dy = sin(phi)
        xdata[particle] += dx
        ydata[particle] += dy
    plt.Circle((circle_point[0], circle_point[1]), circle_radius)
    ln1.set_data(xdata, ydata)
    return ln1,


'''
def updateP2(frame):
    N = 0
    for particle in range(num_particles):
        xdata[particle] += dx
        ydata[particle] += dy

    ln.set_data(xdata, ydata)
    return ln2,
'''

ani = FuncAnimation(figure, update_p1, frames=np.linspace(0, num_steps, num_steps),
                    init_func=init_p1, interval=2000, blit=True, repeat=False)

plt.show()
