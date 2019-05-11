# create generic function data to feed into regression tools

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def sine_fun_for_talos(plot=False):
    x_range = np.arange(-1, 1, 0.1)

    [x1, x2] = np.array([(a, b) for a in x_range for b in x_range]).T

    f = [np.cos(np.pi*a/2)*np.cos(np.pi*b/2)
         for a in x_range for b in x_range]  # *np.exp(-(a**2 + b**2))

    if plot:
        plt.figure(figsize=(10, 8))
        ax = plt.axes(projection='3d')

        ax.scatter3D(x1, x2, f, c=f, cmap='Greens')
        ax.set_xlabel('Independent Variable 1')
        ax.set_ylabel('Independent Variable 2')
        ax.set_zlabel('Dependent Variable')
        ax.set_title(r'Plot of $cos(2\pi x_1) cos(2\pi x_2) e^{-(x_1^2 + x_2^2)}$')
    #    plt.savefig('genericfnc_3Dplot.pdf')
        plt.show()

    x_talos = [[x1[i], x2[i]] for i in range(len(x1))]
    return np.array(x_talos), np.array(f).reshape(len(f),1)


def ski_hill_plot(plot=False):
    x_range = np.arange(-1, 1, 0.01)

    [x1, x2] = np.array([(a, b) for a in x_range for b in x_range]).T

    f = [np.exp(-np.pi*abs(a)/2)*np.exp(-np.pi*abs(b)/2)
         for a in x_range for b in x_range]

    if plot:
        plt.figure(figsize=(10, 8))
        ax = plt.axes(projection='3d')

        ax.scatter3D(x1, x2, f, c=f, cmap='Greens')
        # ax.set_xlabel('$x_1$ (units of distance)')
        # ax.set_ylabel('$x_2$ (units of distance)')
        # ax.set_zlabel('Height of Mountain (units of distance)')
        ax.axis('off')
        ax.set_title(r'3D Rendering of Ski Hill')
        plt.savefig('skihill_3Dplot.pdf')
        plt.show()

    x_talos = [[x1[i], x2[i]] for i in range(len(x1))]
    return np.array(x_talos), np.array(f).reshape(len(f), 1)


def ski_hill_speed_plot(plot=False, save=False):
    x_range = np.arange(-1, 1, 0.02)

    [x1, x2] = np.array([(a, b) for a in x_range for b in x_range]).T

    f = [1-(np.cos(np.pi*a/2)*np.cos(np.pi*b/2))
         for a in x_range for b in x_range]
    noise = np.random.normal(0, 0.03, len(f))
    deltas = [0.3*np.sin(np.pi*i/40) for i in range(40)]
    noise[300:340] = [x - y for x, y in zip(noise[300:340], deltas)]
    f = [f[i]+noise[i] for i in range(len(noise))]

    if plot:
        plt.figure(figsize=(10, 8))
        ax = plt.axes(projection='3d')

        ax.scatter3D(x1, x2, f, c=f, cmap='Greens')
        ax.set_xlabel('$x_1$ (units of distance)')
        ax.set_ylabel('$x_2$ (units of distance)')
        ax.set_zlabel('Skier Speed (units of speed))')
        ax.set_title(r'Average Speed of Skiers on Ski Hill')
        if save:
            plt.savefig('skihill_speed_3Dplot.pdf')
        plt.show()

    x_talos = [[x1[i], x2[i]] for i in range(len(x1))]
    return np.array(x_talos), np.array(f).reshape(len(f), 1)

from tensorflow import keras

