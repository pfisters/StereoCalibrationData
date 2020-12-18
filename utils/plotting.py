import matplotlib.pyplot as plt

def plot_points(points, gt = None):
    plt.plot(points[:,0], points[:,1], 'o', color='red')
    if gt is not None:
        plt.plot(gt[:,0], gt[:,1], '^', color='blue')
    plt.show()