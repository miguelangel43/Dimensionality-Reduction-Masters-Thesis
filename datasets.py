import yaml
import numpy as np

config = yaml.safe_load(open("config.yml"))

class coil2000:
    """https://archive.ics.uci.edu/ml/datasets/Insurance+Company+Benchmark+(COIL+2000)"""

    def __init__(self):
        self.train = np.genfromtxt(config['DATAFOLDER_PATH']+'coil2000/preprocessed_data.txt', delimiter=',')
        self.eval_X = np.genfromtxt(config['DATAFOLDER_PATH']+'coil2000/ticeval2000.txt', delimiter='\t')
        self.eval_y = np.genfromtxt(config['DATAFOLDER_PATH']+'coil2000/tictgts2000.txt', delimiter='\t')
        self.col_names = open(config['DATAFOLDER_PATH']+'coil2000/new_col_names.txt', "r").read().split('\n')

