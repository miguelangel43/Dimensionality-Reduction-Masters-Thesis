from sklearn.model_selection import train_test_split
import yaml
import numpy as np
from PIL import Image

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


class orl:
    """https://cam-orl.co.uk/facedatabase.html"""

    def __init__(self):
        self.X = None
        self.y = None
        self.load_data()

    def load_data(self):
        # Run this cell only one time
        D = np.zeros(shape=(10304, ), dtype=np.int8)
        y = np.array([])

        for i in range(1, 41):
            for j in range(1, 11):
                img = Image.open(
                    f'/Users/espina/Unsynced/Datasets/TFM_datasets/ORL/raw/s{i}/{j}.pgm')
                data = np.asarray(img).reshape(-1)
                D = np.vstack((D, data))
                y = np.append(y, i)

        y = y.astype('int8')
        D = D[1:]

        self.X = D
        self.y = y
