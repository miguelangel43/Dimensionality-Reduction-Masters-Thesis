from sklearn.model_selection import train_test_split
import yaml
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler


config = yaml.safe_load(open("config.yml"))


class coil2000:
    """https://archive.ics.uci.edu/ml/datasets/Insurance+Company+Benchmark+(COIL+2000)"""

    def __init__(self):
        self.data = np.genfromtxt(
            config['DATAFOLDER_PATH']+'coil2000/preprocessed_data.txt', delimiter=',')
        self.col_names = open(
            config['DATAFOLDER_PATH']+'coil2000/new_col_names.txt', "r").read().split('\n')
        self.X = self.data[:, :-1]
        self.y = self.data[:, -1]

        # Balance the dataset using Random Oversampling
        X, y = self.X, self.y
        X = X.reshape(X.shape[0], -1)
        ros = RandomOverSampler(random_state=42)

        X, y = ros.fit_resample(X, y)
        idx = np.random.choice(len(X), size=1000, replace=False)
        self.X = X[idx]
        self.y = y[idx]


class orl:
    """https://cam-orl.co.uk/facedatabase.html"""

    def __init__(self):
        self.X = None
        self.y = None
        self.load_data()

    def load_data(self):
        D = np.zeros(shape=(10304, ), dtype=np.int8)
        y = np.array([])

        for i in range(1, 41):
            for j in range(1, 11):
                img = Image.open(
                    config['DATAFOLDER_PATH'] + f'ORL/raw/s{i}/{j}.pgm')
                data = np.asarray(img).reshape(-1)
                D = np.vstack((D, data))
                y = np.append(y, i)

        y = y.astype('int8')
        D = D[1:]

        self.X = D
        self.y = y - 1


class wine:
    """https://archive.ics.uci.edu/dataset/186/wine+quality"""

    def __init__(self):
        df = pd.read_csv(config['DATAFOLDER_PATH'] +
                         'wine_quality/raw/winequality-red.csv', sep=';')
        df = df.sample(1000)
        self.X = df.drop(labels='quality', axis=1).dropna().to_numpy()
        self.y = df['quality'].to_numpy()


class nba:
    """https://www.kaggle.com/datasets/vivovinco/nba-player-stats"""

    def __init__(self):
        self.X = None
        self.y = None
        self.col_names = None
        self.load_data()

    def load_data(self):
        df = pd.read_csv(config['DATAFOLDER_PATH'] +
                         'nba_players/nba_raw/2021-2022 NBA Player Stats - Regular.csv', sep=';')
        df = df[df['Pos'].isin(['C', 'PF', 'SG', 'PG', 'SF'])]
        label_encoder = LabelEncoder()
        self.y = label_encoder.fit_transform(df[['Pos']])
        X_cols = ['Age',
                  'G',
                  'GS',
                  'MP',
                  'FG',
                  'FGA',
                  'FG%',
                  '3P',
                  '3PA',
                  '3P%',
                  '2P',
                  '2PA',
                  '2P%',
                  'eFG%',
                  'FT',
                  'FTA',
                  'FT%',
                  'ORB',
                  'DRB',
                  'TRB',
                  'AST',
                  'STL',
                  'BLK',
                  'TOV',
                  'PF',
                  'PTS']
        self.X = df[X_cols].dropna().to_numpy()
        self.col_names = [*X_cols, 'Pos']


class fifa:
    """https://www.kaggle.com/datasets/stefanoleone992/fifa-22-complete-player-dataset

    Scraped from https://sofifa.com/"""

    def __init__(self):
        self.X = None
        self.y = None
        self.col_names = None
        self.load_data()

    def load_data(self):
        # Run this cell only one time
        df = pd.read_csv(
            config['DATAFOLDER_PATH'] + 'fifa/players_22.csv')

        X_cols = ['attacking_crossing',
                  'attacking_finishing',
                  'attacking_heading_accuracy',
                  'attacking_short_passing',
                  'attacking_volleys',
                  'skill_dribbling',
                  'skill_curve',
                  'skill_fk_accuracy',
                  'skill_long_passing',
                  'skill_ball_control',
                  'movement_acceleration',
                  'movement_sprint_speed',
                  'movement_agility',
                  'movement_reactions',
                  'movement_balance',
                  'power_shot_power',
                  'power_jumping',
                  'power_stamina',
                  'power_strength',
                  'power_long_shots',
                  'mentality_aggression',
                  'mentality_interceptions',
                  'mentality_positioning',
                  'mentality_vision',
                  'mentality_penalties',
                  'mentality_composure',
                  'defending_marking_awareness',
                  'defending_standing_tackle',
                  'defending_sliding_tackle']

        defender_pos = ["CB",
                        "LB",
                        "RB",
                        "RWB",
                        "LWB",
                        "CB,RB",
                        "CB,LB",
                        "CB,CDM",
                        "LB,RB",
                        "LB,LM",
                        "RB,RM",
                        "LB,LWB",
                        "RB,RWB",
                        "CDM,RB",
                        "CB,CM",
                        "CM,RB",
                        "CM,LB",
                        "LB,LW",
                        "CB,RM",
                        "CB,LWB",
                        "CDM,LB",
                        "RB,RW",
                        "LB,RM",
                        "CAM,RB",
                        "CB,RWB",
                        "CB,CDM,CM",
                        "CB,LB,RB",
                        "LB,LM,LWB",
                        "RB,RM,RWB",
                        "CB,CDM,RB",
                        "CDM,CM,RB",
                        "LB,RB,RM",
                        "LM,LW,RW",
                        "CB,RB,RM",
                        "LM,RB,RM",
                        "CM,LB,LM",
                        "LB,LM,RM",
                        "CB,LB,LWB",
                        "CB,LB,LM",
                        "LB,LM,LW",
                        "CDM,LB,RB",
                        "LB,LM,RB",
                        "CB,RB,RWB",
                        "CB,CDM,LB",
                        "LB,RB,RWB",
                        "CDM,CM,LB",
                        "CDM,RB,RM",
                        "LM,RM,RWB",
                        "RB,RM,RW",
                        "LB,LWB,RB",
                        "CDM,LB,LM",
                        "RB,RM,ST",
                        "CM,LB,RB",
                        "CB,CM,RB",
                        "CAM,LB,LM",
                        "CDM,RB,RWB",
                        "LB,LW,RB",
                        "CB,CM,LB",
                        "LB,RB,RW",
                        "CM,RB,RWB",
                        "CDM,LB,LWB",
                        "CB,LM,LWB",
                        "LWB,RB,RWB",
                        "LM,RB,RWB",
                        "LB,LW,LWB",
                        "CM,LB,LWB",
                        "CB,LWB,RWB",
                        "CB,LM,RB",
                        "CB,LB,RM",
                        "CB,CM,RWB",
                        "CAM,CB,RB"]

        midfielder_pos = ["CM",
                          "RM",
                          "LM",
                          "CAM",
                          "CDM",
                          "CDM,CM",
                          "LM,RM",
                          "CAM,CM",
                          "LWB,RM",
                          "CAM,LM",
                          "CAM,RM",
                          "RM,RW",
                          "CM,RM",
                          "LM,LW",
                          "CM,LM",
                          "CDM,RWB",
                          "CDM,RM",
                          "CAM,LW",
                          "CAM,RW",
                          "CF,RM",
                          "CF,LM",
                          "LW,RM",
                          "LM,RW",
                          "CM,LWB",
                          "CF,CM",
                          "CDM,LM",
                          "LW,LWB",
                          "LM,RWB",
                          "CAM,CDM",
                          "LW,RM,RW",
                          "CF,LM,RM",
                          "CAM,CM,LW",
                          "CAM,RM,RW",
                          "CM,RW",
                          "RM,RWB",
                          "LM,LWB",
                          "CM,RWB",
                          "CM,LW",
                          "CM,LW,RW",
                          "CAM,LM,LW",
                          "CAM,CF,RM",
                          "CAM,CF,LM",
                          "LM,LWB,RM",
                          "LM,RM,RW",
                          "CAM,LW,RW",
                          "CAM,CM,RW",
                          "LM,LW,RM",
                          "CM,RB,RM",
                          "CDM,CM,RM",
                          "CAM,CM,RM",
                          "CDM,CM,LM",
                          "CM,LM,LW",
                          "LWB,RWB",
                          "CM,RM,RW",
                          "CM,LM,ST",
                          "RM,RWB,ST",
                          "RM,RW,RWB",
                          "CF,CM,RM",
                          "CAM,LM,RW",
                          "LM,LW,LWB",
                          "CM,RM,ST",
                          "CM,RM,RWB",
                          "CM,LM,RW",
                          "CM,LM,LWB",
                          "CAM,RM,RWB",
                          "LM,LWB,RWB",
                          "CAM,CM,RM",
                          "CM,LM,RM",
                          "CDM,LM,RM",
                          "CDM,CM,RWB",
                          "CDM,CM,LW",
                          "CAM,LW,RM",
                          "CAM,LM,LWB",
                          "CAM,CDM,RM",
                          "CAM,CDM,LM",
                          "LWB,RM,RWB",
                          "CAM,CDM,CM",
                          "CAM,LM,RM",
                          "LM,RWB,ST",
                          "CM,LWB,RWB",
                          "CM,LW,RM",
                          "CM,LW,LWB",
                          "CF,CM,LM",
                          "CAM,CM,LM",
                          "CDM,LM,LWB",
                          "CDM,LM,LW",
                          "CDM,CM,ST",
                          "CDM,CM,RW",
                          "CDM,CM,LWB",
                          "CDM,CF,CM"]

        attacker_pos = ["ST",
                        "LW",
                        "RW",
                        "CF",
                        "RM,ST",
                        "LW,RW",
                        "LM,ST",
                        "CAM,ST",
                        "CF,RW",
                        "LW,ST",
                        "RW,ST",
                        "CF,ST",
                        "CF,LW",
                        "CM,ST",
                        "CAM,CF",
                        "LW,RW,ST",
                        "LM,RM,ST",
                        "LM,LW,ST",
                        "CAM,LM,ST",
                        "RM,RW,ST",
                        "CAM,CF,ST",
                        "CAM,RM,ST",
                        "CAM,LW,ST",
                        "CAM,CM,ST",
                        "CF,RW,ST",
                        "CF,LW,ST",
                        "CF,LW,RW",
                        "CAM,RW,ST",
                        "CAM,CF,CM",
                        "CAM,CF,LW",
                        "CF,LM,ST",
                        "CAM,CF,RW",
                        "CF,LM,LW",
                        "CF,RM,RW",
                        "LM,RW,ST",
                        "CF,RM,ST",
                        "LW,RM,ST",
                        "CF,CM,ST",
                        "CM,LW,ST",
                        "CF,CM,LW",
                        "LW,LWB,ST",
                        "CM,RW,ST",
                        "CF,LWB,ST",
                        "CF,LM,RW",
                        "CDM,LW,ST"]

        df.loc[df['player_positions'].isin(
            attacker_pos), 'position'] = 'Attacker'
        df.loc[df['player_positions'].isin(
            midfielder_pos), 'position'] = 'Midfielder'
        df.loc[df['player_positions'].isin(
            defender_pos), 'position'] = 'Defender'

        df = df[[*X_cols, 'position']].dropna()

        # df = df[~df.player_positions.isin(['SUB', 'RES'])]
        # df = df.sample(1000)

        self.X = df[X_cols].to_numpy()
        # Assuming 'category_column' is the categorical column in your DataFrame
        label_encoder = LabelEncoder()
        self.y = label_encoder.fit_transform(df[['position']])

        self.col_names = [*X_cols, 'position']


class Income:
    """https://archive.ics.uci.edu/dataset/20/census+income

    488415 x 15
    The dataset contains categorical columns that have to be encoded. It is unbalanced,
    24,720 instances where Y = "<=50K" and 7,841 where Y = ">50K".
    """

    def __init__(self):
        self.col_names = [
            'age',
            'workclass',
            'fnlwgt',
            'education',
            'education-num',
            'marital-status',
            'occupation',
            'relationship',
            'race',
            'sex',
            'capital-gain',
            'capital-loss',
            'hours-per-week',
            'native-country',
            'income'
        ]
        self.data = pd.read_csv(
            config['DATAFOLDER_PATH'] + 'census_income/adult.data', names=self.col_names)
        self.load_data()
        data_np = self.data.to_numpy()
        self.X = data_np[:, :-1]
        self.y = data_np[:, -1]

    def load_data(self):
        # Preprocessing
        # Encoding Categorical Variables
        categorical_cols = ['workclass', 'education', 'marital-status',
                            'occupation', 'relationship', 'race', 'sex', 'native-country', 'income']

        for col in categorical_cols:
            label_encoder = LabelEncoder()
            self.data[col] = label_encoder.fit_transform(self.data[col])
