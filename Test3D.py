import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.animation import FuncAnimation
from math import sqrt, pi
from random import uniform
import random


# Animation Parameters
repeat = False #For animation
interval = 500
box_size = 50000

# Simulation Parameters
np.random.seed(235253)  # Fixing random state for reproducibility
num_particles = 30
num_steps = 50
d = 1  # Angstrom
P = 101325  # Standard pressure in pascal
T = 273.15  # Standard temperature in kelvin
R = 8.314  # Gas constant m^3 Pa K^-1 mol^-1
Na = 6.022 * 10 ** 23  # Avogadro's number

# Derived Parameters
mean_free_path = (R * 10 ** 30) * T / (sqrt(2) * pi * (d ** 2) * Na * P)


def r_calculation(x_data, y_data):
    """
    used to calculate the r coefficient given x_data and y_data
    :param x_data: the x data
    :param y_data: the y data
    :return: the r coefficient of the data
    """
    x_mean = np.mean(x_data)
    y_mean = np.mean(y_data)
    x_std = np.std(x_data, ddof=1)  # ddof = 1, as it is a sample not a population
    y_std = np.std(y_data, ddof=1)

    num_points = len(x_data)
    zx = [(x_data[i] - x_mean) / x_std for i in range(num_points)]
    zy = [(y_data[i] - y_mean) / y_std for i in range(num_points)]

    r = np.sum(np.multiply(zx, zy)) / (num_points - 1)
    return r


def get_component_distances(dist):
    """
    generates random component displacements given a hypotenuse
    :param dist: the hypotenuse
    :return: the component displacements
    """
    dx = uniform(0, dist) * (1 if np.random.random() < 0.5 else -1)
    dy = uniform(0, sqrt((dist ** 2) - (dx ** 2))) * (1 if np.random.random() < 0.5 else -1)
    dz = sqrt((dist ** 2) - (dx ** 2) - (dy ** 2)) * (1 if np.random.random() < 0.5 else -1)
    return dx, dy, dz


def test_component_distances_distribution():
    """
    a test function that verifies that the generation of random lines follow a uniform distribution.
    :return: the average x, y, z of the random lines which should be close to 0.
    """
    dist = mean_free_path
    avgX, avgY, avgZ = get_component_distances(dist)
    for index in range(2, 1000000):
        dx, dy, dz = get_component_distances(dist)
        avgX = (avgX * (index - 1) + dx) / index
        avgY = (avgY * (index - 1) + dy) / index
        avgZ = (avgZ * (index - 1) + dz) / index
    return str(avgX) + " " + str(avgY) + " " + str(avgZ)


def get_random_line(dims=3):
    """
    :param dims: the number of dimensions
    :return: a list of 3 arrays (representing each dimension) where the index of each step is a position for the particle
    """
    lineData = np.zeros((dims, num_steps))
    for index in range(1, num_steps):
        dist = mean_free_path
        dx, dy, dz = get_component_distances(dist)
        lineData[:, index] = lineData[:, index - 1] + [dx, dy, dz]
    return lineData


def calculate_average_distances(data):
    """
    :param data: the random lines data
    :return: the average distance of all particles to origin at each step
    """
    distances_from_origin = [[sqrt(p[0][step] ** 2 + p[1][step] ** 2 + p[2][step] ** 2) for step in range(num_steps)]
                             for p in data]
    return [sum(p) / num_particles for p in zip(*distances_from_origin)]


def calculate_average_distances_squared(average_distances):
    '''
    :param average_distances: the average distance of all particles to origin at each step
    :return: the square of the average distance of all particles to origin at each step
    '''
    return [d ** 2 for d in average_distances]


def update_p1(step_count, dataLines, plt_lines):
    """
    animates the 3d graph
    :param step_count: the current step
    :param dataLines: the data of all the lines
    :param plt_lines: the pointers to the plot lines
    :return: the updated plot lines
    """
    for line, data in zip(plt_lines, dataLines):
        line.set_data(data[0:2, : step_count])
        line.set_3d_properties(data[2, : step_count])
    return plt_lines


def update_p2(step_count):
    """
    the average distance plot
    :param step_count: the current step
    :return: the updated plot line
    """
    ln2.set_data(steps[:step_count + 1], ydata2[:step_count + 1])
    return ln2,


def update_p3(step_count):
    """
    the average distance squared plot
    :param step_count: the current step
    :return: the updated plot line
    """
    ln3.set_data(steps[:step_count + 1], ydata3[:step_count + 1])
    return ln3,


def init_2D_plot(plot, ylabel, ylim, title):
    """
    Instantiates a plot
    :param plot: the plot to instantiate
    :param ylabel: the ylabel of the plot
    :param ylim: the ylimit of the plot
    :param title: the title of the plot
    """
    plot.set_title(title)
    plot.set_xlabel('Steps')
    plot.set_ylabel(ylabel)
    plot.set_xlim(0, num_steps - 1)
    plot.set_ylim(0, ylim)
    plot.grid()


# Pre calculating all data points
steps = range(0, num_steps)
data = [get_random_line() for index in range(num_particles)]
ydata2 = calculate_average_distances(data)
ydata3 = calculate_average_distances_squared(ydata2)

print("R coefficient for average displacement: " + str(r_calculation(steps, ydata2)))
print("R coefficient for average displacement squared: " + str(r_calculation(steps, ydata3)))

# Creating the figure
fig = plt.figure()

# Setting up plot 1 for particles
p1 = fig.add_subplot(2, 2, 1, projection='3d')
lines = [p1.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

p1.set_title('3D Random Walk Simulation in ($\AA$)\'s')
p1.set_xlabel('X')
p1.set_ylabel('Y')
p1.set_zlabel('Z')
p1.set_xlim3d(-box_size, box_size)
p1.set_ylim3d(-box_size, box_size)
p1.set_zlim3d(-box_size, box_size)

# Setting up plot 2 for average distance of particles
p2 = fig.add_subplot(222)
expectedAverageDistanceLine = plt.Arrow(0, 0, num_steps, sqrt(num_steps) * mean_free_path)
p2.add_patch(expectedAverageDistanceLine)
ln2, = p2.plot([], [], color='green', marker='.', linestyle='solid')
init_2D_plot(p2, 'Average Distance ($\AA$)',  1.1 * sqrt(num_steps) * mean_free_path, 'Average Distance at Each Step')

# Setting up plot 3 for average distance squared of particles
p3 = fig.add_subplot(223)
expectedAverageDistanceSquaredLine = plt.Arrow(0, 0, num_steps, num_steps * mean_free_path ** 2)
p3.add_patch(expectedAverageDistanceSquaredLine)
ln3, = p3.plot([], [], color='blue', marker='.', linestyle='solid')
init_2D_plot(p3, 'Average Distance Squared', 1.1 * num_steps * mean_free_path ** 2, 'Average Distance Squared at Each Step')

# Creating the Animation objects
ani1 = FuncAnimation(fig, update_p1, frames=num_steps, fargs=(data, lines), interval=interval, blit=True, repeat=repeat)

ani2 = FuncAnimation(fig, update_p2, frames=num_steps, interval=interval, blit=True, repeat=repeat)

ani3 = FuncAnimation(fig, update_p3, frames=num_steps, interval=interval, blit=True, repeat=repeat)

plt.show()
