# %%
import yaml
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from slmvp import SLMVPTrain, SLMVP_transform
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from datasets import coil2000

data = coil2000()

X_train, y_train = data.train[:, :-1], data.train[:, -1]
# Standardize
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# y_train = scaler.fit_transform(y_train.reshape(-1, 1))

"""
rank: 5, 10, 50, 100, 200
typeK: radial, linear, polynomial
    if typeK == radial
        gamma: 0.01, 0.1, 1, 10, 1/(X_train.shape[0] * X_train.var())
    if typeK == linear
        polyValue: 5
"""
rank = [x for x in [5, 10, 50, 100, 200] if x <= X_train.shape[0]]
typeK = ['radial', 'linear', 'polynomial']  # polynomial
gamma = [0.01, 0.1, 1, 10, 1/(X_train.shape[0] * X_train.var())]
poly_order = 5

kernel_sett = [*[(x, y) for x in ['radial']
                 for y in gamma], *[(x, None) for x in typeK[1:]]]
settings = [(x, *y) for x in rank for y in kernel_sett]


# settings = [(5, 'radial', 1)]
for sett in settings:
    print(sett)
    BAux, Sx = SLMVPTrain(X=X_train.T, Y=y_train,
                          rank=sett[0],
                          typeK=sett[1],
                          gammaX=sett[2],
                          gammaY=sett[2],
                          polyValue=poly_order)

# %%
# ------------------- Model -------------------

# param_grid = dict(rank=[5,10,50,100,200])
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)


# ---------------- Explainability -----------------

# Get the principal components
P_data = SLMVP_transform(BAux.T, X_train.T)

# Load the data into a Pandas df
df = pd.DataFrame(X_train, columns=data.col_names[:-1])

# Add the principal components as columns
for i in range(len(P_data)):
    df['prin'+str(i+1)] = P_data[i]

# Correlations between the original data and each principal component
df_corr = df.corr().iloc[:len(X_train[0]), len(X_train[0]):]


def highlight_cells(val):
    condition = abs(val) >= 0.4
    color = 'yellow' if condition else ''
    font_color = 'black' if condition else ''
    return 'background-color: {}; color: {}'.format(color, font_color)


df_corr.style.applymap(highlight_cells)
# sns.heatmap(df.corr().iloc[:len(X_train[0]), len(X_train[0]):])
# %%
# 1. Examine the eigenvalues to determine how many principal components to consider
eigenvalues = Sx
plt.plot(eigenvalues)
plt.xticks(range(1, 5))
plt.ylabel('eigenvalue')
plt.show()
# %%
# 2. Next, we can compute the principal component scores using the eigenvectors.
# 3. Let's look at the coefficients for the principal components.Because the data
# are standardized, the relative magnitude of each coefficient can be directly
# assessed within a column.
# Too many features, just take highest 5 and lowest 5 in absolute value?
evec_df = pd.DataFrame(BAux.T, columns=data.col_names)


def highlight_cells(val):
    condition = abs(val) >= 0.001
    color = 'yellow' if condition else ''
    font_color = 'black' if condition else ''
    return 'background-color: {}; color: {}'.format(color, font_color)


evec_df.transpose().style.applymap(highlight_cells)
