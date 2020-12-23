import os
import numpy as np
from absl import logging
from utils.plotting import plot_points

class LogAnalyser:
    def __init__(self, path):
        self.log_path = path
        self.scen_path = os.path.join(self.log_path, '0-scenario.txt')
        self.param_path = os.path.join(self.log_path, '0-parameters.txt')
        self.gt_path = os.path.join(self.log_path, 'gt')
        self.rl_path = os.path.join(self.log_path, 'est')
        self.params = self.read_parameters()

    def read_parameters(self):
        params = {}
        for line in open(self.param_path, 'r').readlines():
            key, value = line.split('\t')
            params[key] = value.rstrip()
        return params

    def load(self, identifier : str):
        gts = []
        # logging.info('Loading parameters from %s' % self.gt_path)
        for filename in os.listdir(self.gt_path):
            if filename.startswith(identifier):
                gt = {
                    'it' : int(filename.split('_')[1]),
                    'pt' : np.genfromtxt(os.path.join(self.gt_path, filename), delimiter = '\t', skip_header= 0)
                }
                replicas = min(int(self.params['BUFFER SIZE']) + 1, gt['it'] + 1) 
                gt['pt'] = np.tile(gt['pt'], (replicas, 1))
                gts.append(gt)
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
    
    def load_extrinsics(self, identifier : str = None, items : int = None):
        ex = 'extrinsics.txt'
        gt_extrinsics = np.genfromtxt(os.path.join(self.gt_path, ex), delimiter='\t', skip_header=0)
        rl_extrinsics = np.genfromtxt(os.path.join(self.rl_path, ex), delimiter='\t', skip_header=1)

        rows, cols = rl_extrinsics.shape
        if items is not None:
            rows = min(items, rows)

        if identifier is None:
            rl = rl_extrinsics[:rows, :]
            gt = gt_extrinsics[:rows, :]
        if identifier is 'rms':
            rl = rl_extrinsics[:rows, 1].reshape(rows, 1)
            gt = np.zeros((rows, 1))
        if identifier is 'refrms':
            rl = rl_extrinsics[:rows, 2].reshape(rows, 1)
            gt = np.zeros((rows, 1))
        if identifier is 'tx':
            rl = rl_extrinsics[:rows, 3].reshape(rows, 1)
            gt = gt_extrinsics[:rows, 1].reshape(rows, 1)
        if identifier is 'ty':
            rl = rl_extrinsics[:rows, 4].reshape(rows, 1)
            gt = gt_extrinsics[:rows, 2].reshape(rows, 1)
        if identifier is 'tz':
            rl = rl_extrinsics[:rows, 5].reshape(rows, 1)
            gt = gt_extrinsics[:rows, 3].reshape(rows, 1)
        if identifier is 'rx':
            rl = rl_extrinsics[:rows, 6].reshape(rows, 1)
            gt = gt_extrinsics[:rows, 4].reshape(rows, 1)
        if identifier is 'ry':
            rl = rl_extrinsics[:rows, 7].reshape(rows, 1)
            gt = gt_extrinsics[:rows, 5].reshape(rows, 1)
        if identifier is 'rz':
            rl = rl_extrinsics[:rows, 8].reshape(rows, 1)
            gt = gt_extrinsics[:rows, 6].reshape(rows, 1)
        return gt, rl


    def compare(self, gts : list, pts : list):
        length = min(len(gts), len(pts))
        errors = np.zeros((length, 1))
        for i in range(length):
            gt = gts[i]['pt']
            pt = pts[i]['pt']
            # plot_points(pt, gt)
            errors[i] = np.linalg.norm(np.subtract(gt, pt))
        rms = np.mean(errors)
        logging.info('RMS (mm): %s' % rms)
        return errors
    