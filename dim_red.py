#%%
#main

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import metrics
# ---------------------------------------------------
from slmvp import SLMVPTrain, SLMVP_transform
import pandas as pd
import numpy as np
# ---------------------------------------------------

#X, y = make_classification(n_features=2000, n_redundant=0, n_informative=50, random_state=1, n_clusters_per_class=1, class_sep = 1, n_classes=2)

# ---------------------------------------------------
col_names = ['Climate and Terrain', 'Housing', 'Health Care & Environment',
'Crime','Transportation','Education','The Arts','Recreation','Economics']
df = pd.read_csv('places.txt', sep='\s+', index_col=9,
    names=col_names)
# Perform log transformation
for col in df.columns:
    df[col] = np.log10(df[col])

metric = 'The Arts'
df['y '+'(The Arts)'] = np.where(df[metric]<=df[metric].median(), 0, 1)
df.drop(columns=[metric], inplace=True)

X = df.iloc[:, 0:-1].to_numpy()
y = df.iloc[:, -1].to_numpy()

# ---------------------------------------------------

ranklist = [5, 10, 20, 50, 100, 150]
# ---------------------------------------------------
ranklist = [5]
# ---------------------------------------------------
n_features = X.shape[0]
accuracy_train =[]
accuracy_test = []

#SVM
param_grid_svm = [{'kernel': ['rbf'], 'gamma': [1e-2,1e-3,1e-4],
                     'C': [1, 10, 100]},
                     {'kernel': ['linear'], 'C': [1, 10, 100]}]

for i in ranklist:
    print('Number of dimensions:', i)
    rank = i
    gammaValueX = 1 / (n_features * X.var())
    gammaValueY = 1 / (n_features * y.var())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    #Learn the new space created by SLMVP 
    BAux, Sx = SLMVPTrain(X_train.T, y_train, rank, gammaValueX, gammaValueY) # BAux.T Matrix with eigenvectors?

    #Projection of the Train and Test data
    # ---------------------------------------------------
    P_data = SLMVP_transform(BAux.T,X.T)
    # ---------------------------------------------------
#     P_train = SLMVP_transform(BAux.T,X_train.T)
#     P_test  = SLMVP_transform(BAux.T,X_test.T)

#     #Learning the model
#     clf = GridSearchCV(SVC(),param_grid_svm, cv=3) 
    
#     print('Training')
#     clf.fit(P_train.T, y_train.ravel())
#     grid_results = clf.cv_results_

#     model = clf.best_estimator_

#     #Train Errors
#     y_pred_train = model.predict(P_train.T)
#     accuracy_train.append(metrics.accuracy_score(y_train, y_pred_train))

#     # Test Errors
#     y_pred_test = model.predict(P_test.T)
#     accuracy_test.append(metrics.accuracy_score(y_test, y_pred_test))

# print('Train classification accuracy per number of dimensions 5, 10, 20, 50, 100, 150')
# print(accuracy_train)
# print('Test classification accuracy per number of dimensions 5, 10, 20, 50, 100, 150')
# print(accuracy_test)

#%%
""" 
Step 3: To interpret each component, we must compute the correlations between the original 
data and each principal component.

These correlations are obtained using the correlation procedure. In the variable statement 
we include the first three principal components, "prin1, prin2, and prin3", in addition to 
all nine of the original variables. We use the correlations between the principal components 
and the original variables to interpret these principal components.

Because of standardization, all principal components will have mean 0. The standard deviation 
is also given for each of the components and these are the square root of the eigenvalue.

The correlations between the principal components and the original variables are copied into 
the following table for the Places Rated Example. You will also note that if you look at the 
principal components themselves, then there is zero correlation between the components.
"""

# The correlation between the all the 'training' data on the original axis,
# and the data on the axis

# P_data: the data projected onto the new axis 

prin1 = P_data[0]
prin2 = P_data[1]
prin3 = P_data[2]

eigenvalues = np.diag(Sx)
eigenvector_1 = BAux.T[0]

df['prin1'] = prin1 
df['prin2'] = prin2
df['prin3'] = prin3

df.corr().iloc[:9, 9:]
#%%
import matplotlib.pyplot as plt
plt.plot(eigenvalues)
plt.xticks(range(ranklist[0]))
plt.ylabel('eigenvalue')
plt.show()
#%%


#%%
