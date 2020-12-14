import enum
import os
import cv2
import math
import base64
import numpy as np
import tqdm
import numpy.random as np_rand
from absl import app, logging
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

class FactorySettings:
    def __init__(self, path):
        self.path = path
        self.parameters = self.get_factory_settings()
    
    def get_factory_settings(self) -> dict:
        s = cv2.FileStorage()
        s.open(self.path, cv2.FileStorage_READ)
        algo = s.getNode('algo')
        calib_encoded = algo.getNode('StereoCalibrationString_base64').string()
        calib_decoded = base64.b64decode(calib_encoded).decode('utf-8')
        s.release()

        decoded_file = open('factory_settings.yaml', 'w')
        decoded_file.write(calib_decoded)
        decoded_file.close()

        parameters = {}
        
        cv_file = cv2.FileStorage('factory_settings.yaml', cv2.FileStorage_READ)
        parameters['K1'] = np.asarray(cv_file.getNode('cameraMat1').mat()).reshape(3,3)
        parameters['K2'] = np.asarray(cv_file.getNode('cameraMat2').mat()).reshape(3,3)
        parameters['D1'] = np.asarray(cv_file.getNode('distortCoef1').mat()).reshape(5,)
        parameters['D2'] = np.asarray(cv_file.getNode('distortCoef2').mat()).reshape(5,)
        parameters['R'] = np.asarray(cv_file.getNode('rotationMat').mat()).reshape(3,3)
        parameters['T'] = np.asarray(cv_file.getNode('transitionMat').mat()).reshape(3,)
        cv_file.release()
        os.remove('factory_settings.yaml')
        
        return parameters

class ChangeType(enum.Enum):
    rotation = 1
    translation = 2

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


    def visualize(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        pts = self.points.reshape((self.points.shape[0], 3))

        xdata, ydata, zdata = pts[:,0], pts[:,1], pts[:,2]
        ax.scatter3D(xdata, ydata, zdata)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()


class Change:
    def __init__(self, change_type : ChangeType, iteration : int, data : list):
        self.change_type = change_type
        self.iteration = iteration
        self.data = data
    def to_string(self):
        return '%s\t%s\t%s' % (self.iteration, self.change_type.name, self.data)
    
class Scenario:
    def __init__(self, sequence_length : int, changes : list, scene : Scene, settings : FactorySettings, image_size : tuple):
        self.sequence_length = sequence_length
        self.scene = scene
        self.image_size = image_size

        # rectify
        self.parameters = settings.parameters
        self.rectify()
        
        # check scenario correctness
        for change in changes:
            assert change.iteration < sequence_length
        # sort changes
        changes.sort(key=lambda change: change.iteration)
        self.scenario = changes

        # create output dir
        self.output_dir = self.create_output_dir()

        # save the scenario
        self.save_scenario()
        

    @staticmethod
    def rotation_matrix_to_euler_angles(R):
        sy = math.sqrt(R[0,0] * R[0,0] + R[1,1] * R[1,1])
        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2,1], R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else:
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0
        return np.array([math.degrees(x),math.degrees(y),math.degrees(z)])

    def save_scenario(self):
        directory = os.path.join(self.output_dir, '0-scenario.txt')
        with open(directory, 'w') as file:
            # scene
            file.write('----- SCENE ----- \n')
            file.write('Scene type:\n\t%s\n' % self.scene.scene_type.name)
            file.write('Scene parameters: \n')
            for key, value in self.scene.scene.items():
                file.write('\t%s\t%s\n' % (key, value))
            file.write('\n----- SCENARIO ----- \n')
            file.write('Sequence length: \n\t%s\n' % self.sequence_length)
            file.write('Image Size: \n\t(%s, %s)\n' % (self.image_size))
            file.write('Changes: \n')
            for change in self.scenario:
                file.write('\t%s\n' % change.to_string())

        file.close()

    def create_output_dir(self) -> str:
        output_dir = 'synth_' + str(self.scene.scene_type.name)
        for change in self.scenario:
            output_dir += '_' + str(change.iteration) + '-' + str(max(change.data))
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return output_dir
    
    def save_images(self, left_img, right_img, iteration):
        l_out_image_name = f'L_{iteration}.png'
        l_out_image_path = os.path.join(self.output_dir, l_out_image_name)
        cv2.imwrite(l_out_image_path, left_img)

        r_out_image_name = f'R_{iteration}.png'
        r_out_image_path = os.path.join(self.output_dir, r_out_image_name)
        cv2.imwrite(r_out_image_path, right_img)

    def save_ground_truth(self, 
        points, 
        left_fished, right_fished,
        left_dist, right_dist, 
        left_corr, right_corr, 
        iteration):

        # ground truth directory
        gt_dir = os.path.join(self.output_dir, 'gt')
        if not os.path.exists(gt_dir):
            os.makedirs(gt_dir)

        # extrinsics
        extrinsics_dir = os.path.join(gt_dir, 'extrinsics.txt')
        extrinsics_file = open(extrinsics_dir, 'a')
        angles = self.rotation_matrix_to_euler_angles(self.parameters['R'])
        extrinsics_file.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % ( iteration, 
            self.parameters['T'][0], self.parameters['T'][1], self.parameters['T'][2],
            angles[0], angles[1], angles[2]))
        extrinsics_file.close()

        # 2d/ 3d points
        l_fish = left_fished.reshape(left_fished.shape[0],2)
        r_fish = right_fished.reshape(right_fished.shape[0],2)
        l_dist = left_dist.reshape(left_dist.shape[0],2)
        r_dist = right_dist.reshape(right_dist.shape[0],2)
        l_corr = left_corr.reshape(left_corr.shape[0],2)
        r_corr = right_corr.reshape(right_corr.shape[0],2)

        np.savetxt(os.path.join(gt_dir, 'points3d_%s' % iteration), points, delimiter="\t")
        np.savetxt(os.path.join(gt_dir, 'l-fish_%s' % iteration), l_fish, delimiter="\t")
        np.savetxt(os.path.join(gt_dir, 'r-fish_%s' % iteration), r_fish, delimiter="\t")
        np.savetxt(os.path.join(gt_dir, 'l-dist_%s' % iteration), l_dist, delimiter="\t")
        np.savetxt(os.path.join(gt_dir, 'r-dist_%s' % iteration), r_dist, delimiter="\t")
        np.savetxt(os.path.join(gt_dir, 'l_%s' % iteration), l_corr, delimiter="\t")
        np.savetxt(os.path.join(gt_dir, 'r_%s' % iteration), r_corr, delimiter="\t")

        # parameters
        cv_file = cv2.FileStorage(os.path.join(gt_dir, 'params_%s' % iteration), cv2.FILE_STORAGE_WRITE)
        cv_file.write('R', self.parameters['R'])
        cv_file.write('T', self.parameters['T'])
        cv_file.write('Q', self.parameters['Q'])
        cv_file.write('K1', self.parameters['K1'])
        cv_file.write('K2', self.parameters['K2'])
        cv_file.write('D1', self.parameters['D1'])
        cv_file.write('D2', self.parameters['D2'])
        cv_file.write('R1', self.parameters['R1'])
        cv_file.write('R2', self.parameters['R2'])
        cv_file.write('P1', self.parameters['P1'])
        cv_file.write('P2', self.parameters['P2'])

        cv_file.release() 

    def rectify(self):
        rectification_params = {
            "cameraMatrix1": self.parameters['K1'],
            "distCoeffs1": self.parameters['D1'],
            "cameraMatrix2": self.parameters['K2'],
            "distCoeffs2": self.parameters['D2'],
            "imageSize": self.image_size,
            "R": self.parameters['R'],
            "T": self.parameters['T'],
            "flags": cv2.CALIB_ZERO_DISPARITY,
            "newImageSize": self.image_size,
            "alpha" : .5
        }
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(**rectification_params)
        self.parameters['R1'] = R1
        self.parameters['R2'] = R2
        self.parameters['P1'] = P1
        self.parameters['P2'] = P2
        self.parameters['Q'] = Q


    def fish_points(self, points, left : bool = True):
        fx = self.parameters['K1'][0,0] if left else self.parameters['K2'][0,0]
        fy = self.parameters['K1'][1,1] if left else self.parameters['K2'][1,1]
        cx = self.parameters['K1'][0,2] if left else self.parameters['K2'][0,2]
        cy = self.parameters['K1'][1,2] if left else self.parameters['K2'][1,2]
        
        f = 0.5 * (fx + fy)
        
        fished_points = np.empty(points.shape, float)
        
        for i in range(len(points)):
            x = points[i, 0, 0]
            y = points[i, 0, 1]
            
            xb = x - cx
            yb = y - cy
            
            if(xb == 0.0 and yb == 0.0):
                fished_points[i,0,0] = cx
                fished_points[i,0,1] = cy
                continue
                
            r = np.sqrt(xb * xb + yb * yb)
            theta = r / f
            r_pr = f * math.atan(theta)
            ratio = r_pr / r
                    
            fished_points[i,0,0] = xb * ratio + cx
            fished_points[i,0,1] = yb * ratio + cy
        
        return fished_points


    def projection(self):
        # project
        left_projection_params = {
            "objectPoints": self.scene.points,
            "rvec": cv2.Rodrigues(np.eye(3))[0],
            "tvec": np.zeros((3,1)),
            "cameraMatrix": self.parameters['K1'],
            "distCoeffs": self.parameters['D1']
        }
        right_projection_params = {
            "objectPoints": self.scene.points,
            "rvec": cv2.Rodrigues(self.parameters['R'])[0],
            "tvec": self.parameters['T'],
            "cameraMatrix": self.parameters['K2'],
            "distCoeffs": self.parameters['D2']
        }
        left_dist, _ = cv2.projectPoints(**left_projection_params)
        right_dist, _ = cv2.projectPoints(**right_projection_params)

        # undistort
        left_undistort_params = {
            "src": left_dist,
            "cameraMatrix": self.parameters['K1'],
            "distCoeffs": self.parameters['D1'],
            "R": self.parameters['R1'],
            "P": self.parameters['P1']
        }
        right_undistort_params = {
            "src": right_dist,
            "cameraMatrix": self.parameters['K2'],
            "distCoeffs": self.parameters['D2'],
            "R": self.parameters['R2'],
            "P": self.parameters['P2']
        }

        left_undist = cv2.undistortPoints(**left_undistort_params)
        right_undist = cv2.undistortPoints(**right_undistort_params)

        # fish points
        left_fished = self.fish_points(left_dist, True)
        right_fished = self.fish_points(right_dist, False)

        return left_fished, right_fished, left_dist, right_dist, left_undist, right_undist

    def create_stereo_images(self, left_image_points, right_image_points):
        #  create empty images and draw points
        left_img = np.zeros(self.image_size[::-1], np.int8)
        right_img = left_img.copy()

        color = (255, 255, 255)
        radius = 2
        thickness = -1
        for i in range(0, left_image_points.shape[0]):
            cv2.circle(left_img, (int(left_image_points[i, 0, 0]), int(left_image_points[i, 0, 1])), radius, color, thickness)
            cv2.circle(right_img, (int(right_image_points[i, 0, 0]), int(right_image_points[i, 0, 1])), radius, color, thickness)
        return left_img, right_img

    def update_parameters(self, change : Change):
        if change.change_type == ChangeType.rotation:
            r = self.scene.get_rotation_matrix(change.data)
            self.parameters['R'] = self.parameters['R'].dot(r)
            
        if change.change_type == ChangeType.translation:
            t = self.scene.get_translation_vector(change.data)
            self.parameters['T'] += t

        self.rectify()


    def generate_sequence(self, ):
        # generate images
        left_fished, right_fished, left_dist, right_dist, left_corr, right_corr = self.projection()
        l_img, r_img = self.create_stereo_images(left_fished, right_fished)
        
        change_idx = 0
        next_change = self.scenario[change_idx]

        for iteration in tqdm.trange(0, self.sequence_length):
            if iteration == int(next_change.iteration):
                # update parameters and projections
                self.update_parameters(next_change)
                left_fished, right_fished, left_dist, right_dist, left_corr, right_corr = self.projection()
                l_img, r_img = self.create_stereo_images(left_fished, right_fished)

                # increment change
                change_idx = min(len(self.scenario) - 1, change_idx + 1)
                next_change = self.scenario[change_idx]

            # save images and ground truth
            self.save_images(l_img, r_img, iteration)
            self.save_ground_truth(
                self.scene.points, 
                left_fished, right_fished, 
                left_dist, right_dist, 
                left_corr, right_corr, iteration)


def main(argv):

    # read factory settings
    factory_settings = FactorySettings('factory_settings.json')

    # create synthetic scene
    logging.info('Create Scene')
    scene = {
        'X' : -500,
        'Y' : -500,
        'Z' : 3000,
        'SCALE' : 500,
        'DISTANCE' : 3000,
        'ROT_X' : 5,
        'ROT_Y' : -4,
        'ROT_Z' : 10,
        'MIN' : 0,
        'MAX' : 2000,
        'POINTS': 500
    }
    pts = Scene(SceneType.cube, scene)
    pts.visualize()

    logging.info('Create Scencario')

    changes = []
    changes += [Change(ChangeType.translation, 20, [10, 0, 0])]
    changes += [Change(ChangeType.translation, 30, [0, 10, 0])]
    changes += [Change(ChangeType.translation, 40, [0 ,0, 10])]
    changes += [Change(ChangeType.rotation, 50, [3, 0, 0])]
    changes += [Change(ChangeType.rotation, 60, [0, 3, 0])]
    changes += [Change(ChangeType.rotation, 70, [0, 0, 3])]

    # create scneario
    scenario = Scenario(100, changes, pts, factory_settings, (752, 480))
    scenario.generate_sequence()


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
