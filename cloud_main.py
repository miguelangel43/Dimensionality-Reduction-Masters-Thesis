from datasets import Income
from models import Dim
from sklearn.model_selection import train_test_split

# Instantiate dim. red. framework
dim = Dim()

# Load dataset
inc = Income()

X_train, X_test, y_train, y_test = train_test_split(
    inc.X, inc.y, test_size=0.1, random_state=33)
dim.col_names = inc.col_names
dim.X_train = X_train
dim.y_train = y_train
dim.X_test = X_test
dim.y_test = y_test

dim.new_dim = dict()

# Obtain dimensional spaces
dim.new_dim = dict()
dim.apply_dim(num_dim=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], tflag=[3, 6, 7, 8])

# Run classifiers on reduced space
dim.apply_clf(models=['SVM', 'XGBoost', 'LDA'])

# Get the  correlations
dim.get_corr_table(num_dim=None, abs=False)

# Get Weights
dim.get_weights()
