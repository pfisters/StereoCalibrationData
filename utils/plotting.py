import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

def plot_points(points1 : np.ndarray, points2 : np.ndarray = None, stacked : bool = False, title : str = None):
    if points2 is not None:
        assert points1.shape == points2.shape
    rows, dims = points1.shape
    t = np.arange(0, rows, 1)
    
    if dims == 1:
        plt.plot(t, points1, 'r-')
        if points2 is not None:
            plt.plot(t, points2, 'b-')

    if dims == 2:
        if stacked:
            plt.plot(t, points1[:,0], 'r-')
            plt.plot(t, points1[:,1], 'm-')
            plt.legend(('X', 'Y'))
        else:
            plt.plot(points1[:,0], points1[:,1], 'r-')
        if points2 is not None:
            if stacked:
                plt.plot(t, points2[:,0], 'b-')
                plt.plot(t, points2[:,1], 'c-')
                plt.legend(('X', 'Y'))
            else:
                plt.plot(points2[:,0], points2[:,1], 'b-')
    
    if dims == 3:
        if stacked:
            plt.plot(t, points1[:,0], 'r-')
            plt.plot(t, points1[:,1], 'm-')
            plt.plot(t, points1[:,2], 'y-')
            plt.legend(('X', 'Y', 'Z'))
        else:
            fig = plt.figure()
            ax = plt.axes(projection = '3d')
            xdata, ydata, zdata = points1[:,0], points1[:,1], points1[:,2]
            ax.scatter(xdata, ydata, zdata, marker='o')

        if points2 is not None:
            if stacked:
                plt.plot(t, points2[:,0], 'b-')
                plt.plot(t, points2[:,1], 'c-')
                plt.plot(t, points2[:,2], 'g-')
                plt.legend(('X', 'Y', 'Z'))
            else:
                xdata, ydata, zdata = points2[:,0], points2[:,1], points2[:,2]
                ax.scatter(xdata, ydata, zdata, marker='^')

        if not stacked:
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
    
    if title is not None:
        plt.title(title)
    plt.show()

def plot_extrinsics(ground_truth : np.ndarray, estimated : np.ndarray):
    assert ground_truth.shape == estimated.shape
    rows, cols = ground_truth.shape
    assert cols == 6
    
    fig, axes = plt.subplots(3,2, sharex=True)
    t = np.arange(0, rows, 1)
    axes[0,0].plot(t, ground_truth[:,0], 'r-', t, estimated[:,0], 'b-')
    axes[1,0].plot(t, ground_truth[:,1], 'r-', t, estimated[:,1], 'b-')
    axes[2,0].plot(t, ground_truth[:,2], 'r-', t, estimated[:,2], 'b-')
    axes[0,1].plot(t, ground_truth[:,3], 'r-', t, estimated[:,3], 'b-')
    axes[1,1].plot(t, ground_truth[:,4], 'r-', t, estimated[:,4], 'b-')
    axes[2,1].plot(t, ground_truth[:,5], 'r-', t, estimated[:,5], 'b-')

    plt.show()

