import os
import numpy as np
from absl import logging

class LogAnalyser:
    def __init__(self, path):
        self.log_path = path
        self.gt_path = os.path.join(self.log_path, 'gt')
        self.rl_path = os.path.join(self.log_path, 'est')


    def load(self, identifier : str):
        gts = []
        # logging.info('Loading parameters from %s' % self.gt_path)
        for filename in os.listdir(self.gt_path):
            if filename.startswith(identifier):
                gts.append({
                    'it' : int(filename.split('_')[1]),
                    'pt' : np.genfromtxt(os.path.join(self.gt_path, filename), delimiter = '\t', skip_header= 0)
                    })
        gts.sort(key = lambda gt: gt['it'])

        pts = []
        # logging.info('Loading parameters from %s' % self.rl_path)
        for filename in os.listdir(self.rl_path):
            if filename.startswith(identifier):
                pts.append({
                    'it' : int(filename.split('_')[1]),
                    'pt' : np.genfromtxt(os.path.join(self.rl_path, filename), delimiter='\t', skip_header=0)  
                })
        pts.sort(key = lambda pt: pt['it'])
        return gts, pts
    
    def compare(self, gts : list, pts : list):
        assert len(gts) == len(pts)
        errors = np.zeros((len(gts), 1))
        for i in range(len(gts)):
            gt = gts[i]['pt']
            pt = pts[i]['pt']
            logging.info(type(gt))
            logging.info(type(pt))
            logging.info(gt.shape)
            logging.info(pt.shape)
            errors[i] = np.linalg.norm(np.subtract(gt, pt))
        return np.mean(errors)