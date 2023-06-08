from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import LocallyLinearEmbedding as LLE
from lol import LOL
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from lpproj import LocalityPreservingProjection as LPP
from slmvp import SLMVPTrain, SLMVP_transform
from xgboost import XGBClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from datetime import datetime
from math import floor, sqrt
import pandas as pd
import numpy as np


class Dim:
    """The output of every classifier is a tuple containing the train and test 
    data projected onto the new dimensions, and the new embeddings

    Key Attributes:
    new_dim - dict. contains a tuple (train, test, embeddings)
    """

    def __init__(self, train=None, test=None, col_names=None):
        if train is not None:
            self.X_train = train[:, :-1]
            self.y_train = train[:, -1]
        if test is not None:
            self.X_test = test[:, :-1]
            self.y_test = test[:, -1]
        self.col_names = col_names
        self.new_dim = dict()  # X_train, X_test, components
        self.scores = dict()
        self.results = dict()
        self.num_dim = None

    def pickle_dim(self, output_path):
        if len(self.new_dim) == 0:
            print('No dimensions loaded.')
            return
        with open('dim/' + output_path + '.pkl', 'wb') as f:
            pickle.dump(self.new_dim, f)

    def unpickle_dim(self, input_path):
        with open(input_path, 'rb') as f:
            self.new_dim = pickle.load(f)

    def get_weights(self):
        """Return the eigenvalues. Calculates the variation of the data projected
        onto the discovered dimensions as a proxy for the eigenvalues."""
        res = pd.DataFrame()
        for key in self.new_dim.keys():
            # Calculate variations
            var_dims = [np.var(self.new_dim[key][0][i])
                        for i in range(len(self.new_dim[key][0]))]
            por_eigenvals = [x/sum(var_dims) for x in var_dims]
            res[key+('Var',)] = var_dims
            res[key+('Var %',)] = por_eigenvals
            # Calculate regression betas
            model = LinearRegression().fit(
                self.new_dim[key][0].T, self.y_train)
            betas = [abs(x) for x in model.coef_]
            r_squared = model.score(
                self.new_dim[key][0].T, self.y_train)
            res[key+('Beta',)] = betas
            res[key+('Beta %',)] = [x/sum(betas) for x in betas]
            res[key+('R^2',)] = [r_squared for x in range(len(betas))]

        res.columns = pd.MultiIndex.from_tuples(
            res.columns.to_list())

        res.to_csv('/Users/espina/Documents/TFM/tfm_code/evalues/' +
                   datetime.now().strftime('%m-%d-%H:%M') + '.csv')
        res.to_excel('/Users/espina/Documents/TFM/tfm_code/evalues/' +
                     datetime.now().strftime('%m-%d-%H:%M') + '.xlsx')

        return res

    def get_corr_table(self, num_dim=None):
        if num_dim is None:
            num_dim = self.num_dim
        # Load the data into a Pandas df
        if self.col_names is not None:
            df = pd.DataFrame(self.X_train, columns=self.col_names[:-1])
        else:
            df = pd.DataFrame(self.X_train)
        for key in tqdm(self.new_dim.keys()):
            for i in range(num_dim):
                df[key+(i,)] = self.new_dim[key][0][i]

        # Correlations between the original data and each principal component
        df_corr = df.corr().iloc[:len(self.X_train[0]), len(self.X_train[0]):]

        # Make df multi-index
        df_corr.columns = pd.MultiIndex.from_tuples(
            df_corr.columns.to_list())

        # Take only the abs value of the correlations
        df_corr = df_corr.abs()

        # Save result in csv
        df_corr.to_csv('/Users/espina/Documents/TFM/tfm_code/corr/corr_' +
                       datetime.now().strftime('%m-%d-%H:%M')+'_'+str(num_dim)+'.csv')
        df_corr.to_excel('/Users/espina/Documents/TFM/tfm_code/corr/corr_' +
                         datetime.now().strftime('%m-%d-%H:%M')+'_'+str(num_dim)+'.xlsx')

        return df_corr

    # 5, 10, 50 dims takes 5min
    def apply_dim(self, num_dim=[1, 2, 5, 10], multilabel=None):
        """Run dim. red. algorithms and save new features in self.new_dim"""
        if not isinstance(num_dim, list):
            self.num_dim = num_dim
            num_dim = [num_dim]

        pbar = tqdm(num_dim)
        for dim in pbar:

            key = (str(dim) + 'Dim', 'SLMVP', 'Polynomial-Order=5')
            pbar.set_description(str(key))
            self.new_dim[key] = self.slmvp_model(
                dim, 'polynomial', poly_order=5, multilabel=multilabel)

            key = (str(dim) + 'Dim', 'SLMVP', 'Linear')
            pbar.set_description(str(key))
            self.new_dim[key] = self.slmvp_model(
                dim, 'linear', multilabel=multilabel)

            key = (str(dim) + 'Dim', 'SLMVP', 'Radial-Gammas=0.01')
            pbar.set_description(str(key))
            self.new_dim[key] = self.slmvp_model(
                dim, 'radial', gammas=0.01, multilabel=multilabel)

            key = (str(dim) + 'Dim', 'SLMVP', 'Radial-Gammas=0.1')
            pbar.set_description(str(key))
            self.new_dim[key] = self.slmvp_model(
                dim, 'radial', gammas=0.1, multilabel=multilabel)

            key = (str(dim) + 'Dim', 'PCA', '')
            pbar.set_description(str(key))
            self.new_dim[key] = self.pca_model(dim)

            # # No known way of getting the components
            # key = (str(dim) + 'Dim', 'KPCA', 'Linear')
            # pbar.set_description(str(key))
            # self.new_dim[key] = self.kpca_model(dim, 'linear')

            # key = (str(dim) + 'Dim', 'KPCA', 'Polynomial-Order=5')
            # pbar.set_description(str(key))
            # self.new_dim[key] = self.kpca_model(dim, 'poly')

            # key = (str(dim) + 'Dim', 'KPCA', 'Radial-Gamma=0.1')
            # pbar.set_description(str(key))
            # self.new_dim[key] = self.kpca_model(dim, 'rbf', gamma=0.1)

            key = (str(dim) + 'Dim', 'KPCA', 'Radial-Gamma=' +
                   str(round(1/len(self.X_train[0]), 3)))
            pbar.set_description(str(key))
            self.new_dim[key] = self.kpca_model(dim, 'rbf')

            if multilabel is None:
                # No known way of getting the components
                key = (str(dim) + 'Dim', 'LOL', '')
                pbar.set_description(str(key))
                self.new_dim[key] = self.lol_model(dim, n_components=dim)

            # k = floor(sqrt(min(len(self.X_train), len(self.X_train[0]))))
            # key = (str(dim) + 'Dim', 'LPP', 'k=' + str(k))
            # pbar.set_description(str(key))
            # self.new_dim[key] = self.lpp_model(dim, k)

            k = floor(sqrt(len(self.X_train)))
            reg = 0.001
            # No known way of getting the components
            key = (str(dim) + 'Dim', 'LLE', 'k=' + str(k) + '-reg=' + str(reg))
            pbar.set_description(str(key))
            self.new_dim[key] = self.lle_model(dim, k, reg)

        # Save the results in a csv, xls
        result_dim = pd.DataFrame()
        for key in self.new_dim.keys():
            result_dim[key] = self.new_dim[key][0][0]
        result_dim.columns = pd.MultiIndex.from_tuples(
            result_dim.columns.to_list())
        result_dim.to_csv('/Users/espina/Documents/TFM/tfm_code/dim/' +
                          datetime.now().strftime('%m-%d-%H:%M') + '.csv')
        result_dim.to_excel('/Users/espina/Documents/TFM/tfm_code/dim/' +
                            datetime.now().strftime('%m-%d-%H:%M') + '.xlsx')
        # Save as pickle
        self.pickle_dim(output_path=datetime.now().strftime(
            '%m-%d-%H:%M'))

    def slmvp_model(self, n, type_kernel, gammas=None, poly_order=None, multilabel=None):
        # Get the principal components
        BAux = SLMVPTrain(X=self.X_train.T, Y=self.y_train,
                          rank=n,
                          typeK=type_kernel,
                          gammaX=gammas,
                          gammaY=gammas,
                          polyValue=poly_order,
                          multilabel=multilabel)

        # Get the data projected onto the new dimensions
        data_train, data_test = SLMVP_transform(
            BAux.T, self.X_train.T), SLMVP_transform(BAux.T, self.X_test.T)

        return data_train, data_test, BAux

    def lpp_model(self, n, k):
        lpp = LPP(n_components=n, n_neighbors=k)
        lpp.fit(self.X_train)
        X_lpp_train = lpp.transform(self.X_train)
        X_lpp_test = lpp.transform(self.X_test)

        return X_lpp_train.T, X_lpp_test.T, lpp.projection_

    def pca_model(self, n):
        pca_model = PCA(n_components=n).fit(self.X_train)
        X_pca_train = pca_model.transform(self.X_train)
        X_pca_test = pca_model.transform(self.X_test)
        # return train, test, eigenvectors, eigenvalues
        return X_pca_train.T, X_pca_test.T, pca_model.components_, pca_model.explained_variance_

    def lle_model(self, n, k, _reg):
        lle = LLE(n_neighbors=k, n_components=n, reg=_reg)
        X_lle_train = lle.fit_transform(self.X_train)
        X_lle_test = lle.transform(self.X_test)

        return X_lle_train.T, X_lle_test.T

    def lda_model(self, n, y_train, X_val='na'):
        lda = LDA(n_components=n)
        X_lda_train = lda.fit_transform(self.X_train, y_train)
        X_lda_test = lda.transform(self.X_test)
        if X_val != 'na':
            X_lda_val = lda.transform(X_val)

        return X_lda_train, X_lda_val, X_lda_test

    def kpca_model(self, n, type_kernel, gamma=None):
        kernel_pca = KernelPCA(
            n_components=n, kernel=type_kernel, fit_inverse_transform=True, gamma=gamma
        )
        X_kpca_train = kernel_pca.fit(self.X_train).transform(self.X_train)
        X_kpca_test = kernel_pca.transform(self.X_test)

        return X_kpca_train.T, X_kpca_test.T

    def lol_model(self, n, n_components):
        lmao = LOL(n_components=n_components, svd_solver='full')
        lmao.fit(self.X_train, self.y_train)
        X_lol_train = lmao.transform(self.X_train)
        X_lol_test = lmao.transform(self.X_test)

        return X_lol_train.T, X_lol_test.T

    def pickle_scores(self, output_path):
        if len(self.scores) == 0:
            print('No scores loaded.')
            return
        with open('scores/' + output_path + '.pkl', 'wb') as f:
            pickle.dump(self.scores, f)

    def apply_clf(self, model):
        """Run classifiers and save new scores in self.scores and in folder /scores"""

        if model == 'XGBoost':
            xgb_pipe = Pipeline([('mms', MinMaxScaler()),
                                ('xgb', XGBClassifier())])
            params = [{'xgb__n_estimators': [5, 10, 20, 50, 100]}]
            gs_xgb = GridSearchCV(xgb_pipe,
                                  param_grid=params,
                                  scoring='accuracy',
                                  cv=5)

            for key_dim in tqdm(self.new_dim.keys(), desc='XGBoost'):
                gs_xgb.fit(self.new_dim[key_dim][0].T, self.y_train)
                self.scores[key_dim] = [
                    gs_xgb.score(self.new_dim[key_dim][1].T, self.y_test),
                    gs_xgb.best_params_
                ]

            # Pickle all the scores
            self.pickle_scores(datetime.now().strftime('%m-%d-%H:%M'))

        elif model == 'Linear Regression':
            reg_pipe = Pipeline([('mms', MinMaxScaler()),
                                ('reg', LinearRegression())])

            for key_dim in tqdm(self.new_dim.keys(), desc='Linear Reg.'):
                reg_pipe.fit(self.new_dim[key_dim][0].T, self.y_train)
                self.scores[key_dim] = [
                    reg_pipe.score(self.new_dim[key_dim][1].T, self.y_test),
                    ''
                ]

            # Pickle all the scores
            self.pickle_scores(datetime.now().strftime('%m-%d-%H:%M'))

        elif model == 'KNN':
            knn_pipe = Pipeline([('mms', MinMaxScaler()),
                                ('knn', KNeighborsClassifier())])
            params = [{'knn__n_neighbors': [3, 5, 10, 20, 50, 100]}]
            gs_knn = GridSearchCV(knn_pipe,
                                  param_grid=params,
                                  scoring='accuracy',
                                  cv=5)

            for key_dim in tqdm(self.new_dim.keys(), desc='KNN'):
                gs_knn.fit(self.new_dim[key_dim][0].T, self.y_train)
                self.scores[key_dim] = [
                    gs_knn.score(self.new_dim[key_dim][1].T, self.y_test),
                    gs_knn.best_params_
                ]

            # Pickle all the scores
            self.pickle_scores(datetime.now().strftime('%m-%d-%H:%M'))

        df = pd.DataFrame.from_dict(self.scores, orient='index', columns=[
                                    'Best Score', 'Params']).reset_index()
        df[['Dimensions', 'Dim. Technique', 'Dim. Params']] = pd.DataFrame(
            df['index'].tolist(), index=df.index)

        df = df.drop('index', axis=1)

        df = df.sort_values('Best Score', ascending=False)

        df.to_csv('/Users/espina/Documents/TFM/tfm_code/scores/' +
                  datetime.now().strftime('%m-%d-%H:%M') + '.csv')
        df.to_excel('/Users/espina/Documents/TFM/tfm_code/scores/' +
                    datetime.now().strftime('%m-%d-%H:%M') + '.xlsx')

        # df = df.groupby('Dim. Technique', as_index=False).first()

        return df

    def plot_artificial(self, n_rows, n_cols, figsize=(15, 12), save_name=None):
        fig, ax = plt.subplots(
            n_rows, n_cols, figsize=figsize)
        y = self.y_train
        for idx, key_dim in enumerate(list(self.new_dim.keys())):
            ax[floor(idx/n_cols)][idx % n_cols].scatter(self.new_dim[key_dim]
                                                        [0][0], self.new_dim[key_dim][0][1], c=y)
            ax[floor(idx/n_cols)][idx % n_cols].set_title(list(self.new_dim.keys())
                                                          [idx][1] + ' ' + list(self.new_dim.keys())[idx][2])
        if save_name is not None:
            plt.savefig(
                '/Users/espina/Documents/TFM/tfm_code/plots/' + save_name + '.png')

    def plot_artificial_3D(self, n_rows, n_cols, figsize=(15, 12), save_name=None):
        fig = plt.figure(figsize=figsize)
        y = self.y_train
        for idx, key_dim in enumerate(list(self.new_dim.keys())):
            ax = fig.add_subplot(n_rows, n_cols, idx+1, projection='3d')
            ax.scatter(self.new_dim[key_dim][0][0],
                       self.new_dim[key_dim][0][1],
                       self.new_dim[key_dim][0][2], c=y)
            ax.set_title(list(self.new_dim.keys())[
                         idx][1] + ' ' + list(self.new_dim.keys())[idx][2])
        if save_name is not None:
            plt.savefig(
                '/Users/espina/Documents/TFM/tfm_code/plots/' + save_name + '.png')

    def plot_artificial_multilabel(self, n_rows, n_cols, figsize=(15, 12), save_name=None):
        fig, ax = plt.subplots(
            n_rows, n_cols, figsize=figsize)

        zero_class = np.where(self.y_train[:, 0])
        one_class = np.where(self.y_train[:, 1])
        for idx, key_dim in enumerate(list(self.new_dim.keys())):
            X = self.new_dim[key_dim][0].T
            # Plot all data points in grey
            p1 = ax[floor(idx/n_cols)][idx % n_cols].scatter(X[:, 0],
                                                             X[:, 1], s=40, c="gray", edgecolors=(0, 0, 0))
            # Plot data points with a 1 in the first position
            p2 = ax[floor(idx/n_cols)][idx % n_cols].scatter(
                X[zero_class, 0],
                X[zero_class, 1],
                s=160,
                edgecolors="b",
                facecolors="none",
                linewidths=2,
                label="Class 1",
            )
            # Plot data points with a 1 in the second position
            p3 = ax[floor(idx/n_cols)][idx % n_cols].scatter(
                X[one_class, 0],
                X[one_class, 1],
                s=80,
                edgecolors="orange",
                facecolors="none",
                linewidths=2,
                label="Class 2",
            )

            ax[floor(idx/n_cols)][idx % n_cols].set_title(list(self.new_dim.keys())
                                                          [idx][1] + ' ' + list(self.new_dim.keys())[idx][2])

            handles, labels = ax[floor(idx/n_cols)][idx %
                                                    n_cols].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right')

        if save_name is not None:
            plt.savefig(
                '/Users/espina/Documents/TFM/tfm_code/plots/' + save_name + '.png')

    def plot_artificial_multilabel_3D(self, n_rows, n_cols, figsize=(15, 12), save_name=None):
        fig = plt.figure(figsize=figsize)
        zero_class = np.where(self.y_train[:, 0])
        one_class = np.where(self.y_train[:, 1])
        for idx, key_dim in enumerate(list(self.new_dim.keys())):
            ax = fig.add_subplot(n_rows, n_cols, idx+1, projection='3d')
            X = self.new_dim[key_dim][0].T
            # Plot all data points in grey
            ax.scatter(X[:, 0], X[:, 1], s=40, c="gray", edgecolors=(0, 0, 0))
            # Plot data points with a 1 in the first position
            ax.scatter(
                X[zero_class, 0],
                X[zero_class, 1],
                s=160,
                edgecolors="b",
                facecolors="none",
                linewidths=2,
                label="Class 1",
            )
            # Plot data points with a 1 in the second position
            ax.scatter(
                X[one_class, 0],
                X[one_class, 1],
                s=80,
                edgecolors="orange",
                facecolors="none",
                linewidths=2,
                label="Class 2",
            )
            ax.set_title(list(self.new_dim.keys())[
                         idx][1] + ' ' + list(self.new_dim.keys())[idx][2])

        if save_name is not None:
            plt.savefig(
                '/Users/espina/Documents/TFM/tfm_code/plots/' + save_name + '.png')
