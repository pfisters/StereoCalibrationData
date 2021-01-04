import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

def line_plot_points(points1 : np.ndarray, points2 : np.ndarray = None, stacked : bool = False, title : str = None, 
    iterations : int = -1,
    marker1 : str = 'r-', marker2 : str = 'b-', marker3 : str = 'm-', 
    marker4 : str = 'c-', marker5 : str = 'y-', marker6 : str = 'g-'):
    if points2 is not None:
        assert points1.shape == points2.shape
    
    if len(points1.shape) == 1:
        rows = points1.shape[0]
        dims = 1
    elif len(points1.shape) == 2:
        rows, dims = points1.shape
    elif len(points1.shape) == 3:
        rows, _, dims = points1.shape
        points1 = points1.reshape(rows, dims)
    
    if iterations > 0:
        rows = min(rows, iterations)

    t = np.arange(0, rows, 1)
    
    if dims == 1:
        plt.plot(t, points1, marker1)
        if points2 is not None:
            plt.plot(t, points2, marker2)

    if dims == 2:
        if stacked:
            plt.plot(t, points1[:,0], marker1)
            plt.plot(t, points1[:,1], marker3)
            plt.legend(('X', 'Y'))
        else:
            plt.plot(points1[:,0], points1[:,1], marker1)
        if points2 is not None:
            if stacked:
                plt.plot(t, points2[:,0], marker2)
                plt.plot(t, points2[:,1], marker4)
                plt.legend(('X', 'Y'))
            else:
                plt.plot(points2[:,0], points2[:,1], marker2)
    
    if dims == 3:
        if stacked:
            plt.plot(t, points1[:,0], marker1)
            plt.plot(t, points1[:,1], marker3)
            plt.plot(t, points1[:,2], marker5)
            plt.legend(('X', 'Y', 'Z'))
        else:
            fig = plt.figure()
            ax = plt.axes(projection = '3d')
            xdata, ydata, zdata = points1[:,0], points1[:,1], points1[:,2]
            ax.scatter(xdata, ydata, zdata, marker='o')

        if points2 is not None:
            if stacked:
                plt.plot(t, points2[:,0], marker2)
                plt.plot(t, points2[:,1], marker4)
                plt.plot(t, points2[:,2], marker6)
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

def scatter_plot_points(points1 : np.ndarray, points2 : np.ndarray = None):
    if len(points1.shape) == 2:
        rows, dims = points1.shape
    if len(points1.shape) == 3:
        rows, _, dims = points1.shape
        points1 = points1.reshape(rows, dims)

    plt.scatter(points1[:,0], points1[:,1])
    plt.show()


def plot_extrinsics(estimated : np.ndarray, ground_truth : np.ndarray = None, iterations : int = -1, title : str = None):
    if ground_truth is not None:
        assert ground_truth.shape == estimated.shape
    rows, cols = estimated.shape
    if iterations > 0:
        rows = min(iterations, rows)
    assert cols == 6
    
    fig, axes = plt.subplots(3,2, sharex=True)
    t = np.arange(0, rows, 1)
    
    if ground_truth is not None:
        axes[0,0].plot(t, ground_truth[:rows,0])
        axes[1,0].plot(t, ground_truth[:rows,1])
        axes[2,0].plot(t, ground_truth[:rows,2])
        axes[0,1].plot(t, ground_truth[:rows,3])
        axes[1,1].plot(t, ground_truth[:rows,4])
        axes[2,1].plot(t, ground_truth[:rows,5])

    axes[0,0].plot(t, estimated[:rows,0], 'b-')
    axes[0,0].set_ylabel('Tx')
    axes[1,0].plot(t, estimated[:rows,1], 'b-')
    axes[1,0].set_ylabel('Ty')
    axes[2,0].plot(t, estimated[:rows,2], 'b-')
    axes[2,0].set_ylabel('Tz')
    axes[2,0].set_xlabel('mm')
    axes[0,1].plot(t, estimated[:rows,3], 'b-')
    axes[0,1].yaxis.set_label_position('right')
    axes[0,1].yaxis.tick_right()
    axes[0,1].set_ylabel('Rx')
    axes[1,1].plot(t, estimated[:rows,4], 'b-')
    axes[1,1].yaxis.set_label_position('right')
    axes[1,1].yaxis.tick_right()
    axes[1,1].set_ylabel('Ry')
    axes[2,1].plot(t, estimated[:rows,5], 'b-')
    axes[2,1].yaxis.set_label_position('right')
    axes[2,1].yaxis.tick_right()
    axes[2,1].set_ylabel('Rz')
    axes[2,1].set_xlabel('degrees')

    if title is not None:
        plt.suptitle(title)
    plt.show()

