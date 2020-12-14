import cv2
import numpy as np
import base64
import os

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