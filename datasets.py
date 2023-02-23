from sklearn.model_selection import train_test_split
import yaml
import numpy as np

config = yaml.safe_load(open("config.yml"))


class coil2000:
    """https://archive.ics.uci.edu/ml/datasets/Insurance+Company+Benchmark+(COIL+2000)"""

    def __init__(self):
        self.data = np.genfromtxt(
            config['DATAFOLDER_PATH']+'coil2000/preprocessed_data.txt', delimiter=',')
        self.train, self.test = train_test_split(
            self.data, test_size=0.20, random_state=42)
        self.col_names = open(
            config['DATAFOLDER_PATH']+'coil2000/new_col_names.txt', "r").read().split('\n')
