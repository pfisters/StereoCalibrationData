import enum
import numpy as np
import numpy.random as np_rand
import matplotlib.pyplot as plt
import math


class SceneType(enum.Enum):
    grid = 1
    cube = 2

class Scene:
    def __init__(self, scene_type : SceneType, scene : dict):
        self.scene_type = scene_type
        self.scene = scene
        self.points = self.generate_grid(self) if scene_type == SceneType.grid else self.generate_cube(self)

    @staticmethod
    def generate_grid(self):
        # create 3D-grid (äquidistant grid at z = 0)
        grid_points = np.zeros((self.scene['X'] * self.scene['Y'], 3), 
            dtype=float)
        grid_points[:, :2] = np.mgrid[0:self.scene['X'], 0:self.scene['Y']].T.reshape(-1, 2)
        grid_points *= self.scene['SCALE']
        grid_points[:, 2] = self.scene['DISTANCE']

        rot = self.get_rotation_matrix([self.scene['ROT_X'], self.scene['ROT_Y'], self.scene['ROT_Z']])
        grid_rot = rot @ grid_points.T
        return grid_rot.T

    @staticmethod
    def generate_cube(self):
        cube_points = np_rand.uniform(
            low=self.scene['MIN'],
            high=self.scene['MAX'],
            size=(self.scene['POINTS'], 3))
        cube_points[:,0] += self.scene['X']
        cube_points[:,1] += self.scene['Y']
        cube_points[:,2] += self.scene['Z']

        rot = self.get_rotation_matrix([self.scene['ROT_X'], self.scene['ROT_Y'], self.scene['ROT_Z']])
        cube_rot = rot @ cube_points.T
        return cube_rot.T

    @staticmethod
    def get_translation_vector(trans : list):
        return np.asarray(trans)
    
    @staticmethod
    def get_rotation_matrix(rot : list):
        x_angle = math.radians(rot[0])
        x_rot_mat = np.array(
            [[1., 0., 0.],
            [0., math.cos(x_angle), (-1. * math.sin(x_angle))],
            [0., math.sin(x_angle), math.cos(x_angle)]])
        
        y_angle = math.radians(rot[1])
        y_rot_mat = np.array(
            [[math.cos(y_angle), 0., math.sin(y_angle)],
            [0., 1., 0.],
            [(-1. * math.sin(y_angle)), 0., math.cos(y_angle)]])
        
        z_angle = math.radians(rot[2])
        z_rot_mat = np.array(
            [[math.cos(z_angle), (-1. * math.sin(z_angle)), 0.],
            [math.sin(z_angle), math.cos(z_angle), 0.],
            [0., 0., 1.]])
        
        return z_rot_mat.dot(y_rot_mat.dot(x_rot_mat))


    def visualize(self, gt = None):
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        pts = self.points.reshape((self.points.shape[0], 3))

        xdata, ydata, zdata = pts[:,0], pts[:,1], pts[:,2]
        ax.scatter3D(xdata, ydata, zdata, marker='o')

        if gt is not None:
            xdata, ydata, zdata = gt[:,0], gt[:,1], gt[:,2]
            ax.scatter3D(xdata, ydata, zdata, marker='^')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
