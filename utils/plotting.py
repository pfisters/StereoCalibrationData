import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

def plot_points(points1 : np.ndarray, points2 : np.ndarray = None):
    assert points1.shape == points2.shape
    rows, dims = points1.shape

    if dims == 1:
        t = np.arange(0, rows, 1)
        plt.plot(t, points1, 'r--')
        if points2 is not None:
            plt.plot(t, points2, 'b--')

    if dims == 2:
        plt.plot(points1[:,0], points1[:,1], 'o', color='red')
        if points2 is not None:
            plt.plot(points2[:,0], points2[:,1], '^', color='blue')
    
    if dims == 3:
        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        xdata, ydata, zdata = points1[:,0], points1[:,1], points1[:,2]
        ax.scatter(xdata, ydata, zdata, marker='o')

        if points2 is not None:
            xdata, ydata, zdata = points2[:,0], points2[:,1], points2[:,2]
            ax.scatter(xdata, ydata, zdata, marker='^')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    plt.show()