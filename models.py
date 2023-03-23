from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import LocallyLinearEmbedding as LLE
from lol import LOL
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from lpproj import LocalityPreservingProjection as LPP
from slmvp import SLMVPTrain, SLMVP_transform
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
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

    def __init__(self, train, test, col_names):
        self.X_train = train[:, :-1]
        self.y_train = train[:, -1]
        self.X_test = test[:, :-1]
        self.y_test = test[:, -1]
        self.col_names = col_names
        self.new_dim = dict()  # X_train, X_test, components
        self.scores = dict()
        self.results = dict()

    def pickle_dim(self, output_path):
        if len(self.new_dim) == 0:
            print('No dimensions loaded.')
            return
        with open('dim/' + output_path + '.pkl', 'wb') as f:
            pickle.dump(self.new_dim, f)

    def unpickle_dim(self, input_path):
        with open(input_path, 'rb') as f:
            self.new_dim = pickle.load(f)

    def get_eigenvalues(self):
        res = pd.DataFrame()
        for key in self.new_dim.keys():
            var_dims = [np.var(self.new_dim[key][0][i])
                        for i in range(len(self.new_dim[key][0]))]
            por_eigenvals = [x/sum(var_dims) for x in var_dims]
            res[key+('Var',)] = var_dims
            res[key+('%',)] = por_eigenvals

        res.columns = pd.MultiIndex.from_tuples(
            res.columns.to_list())

        res.to_csv('/Users/espina/Documents/TFM/tfm_code/evalues/' +
                   datetime.now().strftime('%m-%d-%H:%M') + '.csv')
        res.to_excel('/Users/espina/Documents/TFM/tfm_code/evalues/' +
                     datetime.now().strftime('%m-%d-%H:%M') + '.xlsx')

        return res

    def get_corr_table(self, num_dim):
        # Load the data into a Pandas df
        df = pd.DataFrame(self.X_train, columns=self.col_names[:-1])
        for key in self.new_dim.keys():
            if num_dim == int(key[0][0]):
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

    def apply_dim(self, num_dim=[1, 2, 5, 10]):  # 5, 10, 50 dims takes 5min
        """Run dim. red. algorithms and save new features in self.new_dim"""
        if not isinstance(num_dim, list):
            num_dim = [num_dim]

        pbar = tqdm(num_dim)
        for dim in pbar:
            key = (str(dim) + 'Dim', 'SLMVP', 'Polynomial-Order=5')
            pbar.set_description(str(key))
            self.new_dim[key] = self.slmvp_model(
                dim, 'polynomial', poly_order=5)
            key = (str(dim) + 'Dim', 'SLMVP', 'Linear')
            pbar.set_description(str(key))
            self.new_dim[key] = self.slmvp_model(dim, 'linear')
            key = (str(dim) + 'Dim', 'SLMVP', 'Radial-Gammas=0.01')
            pbar.set_description(str(key))
            self.new_dim[key] = self.slmvp_model(dim, 'radial', gammas=0.01)
            key = (str(dim) + 'Dim', 'SLMVP', 'Radial-Gammas=0.1')
            pbar.set_description(str(key))
            self.new_dim[key] = self.slmvp_model(dim, 'radial', gammas=0.1)
            key = (str(dim) + 'Dim', 'SLMVP', 'Radial-Gammas=1')
            pbar.set_description(str(key))
            self.new_dim[key] = self.slmvp_model(dim, 'radial', gammas=1)
            key = (str(dim) + 'Dim', 'SLMVP', 'Radial-Gammas=10')
            pbar.set_description(str(key))
            self.new_dim[key] = self.slmvp_model(dim, 'radial', gammas=10)
            key = (str(dim) + 'Dim', 'PCA', '')
            pbar.set_description(str(key))
            self.new_dim[key] = self.pca_model(dim)
            # No known way of getting the components
            key = (str(dim) + 'Dim', 'KPCA', 'Linear')
            pbar.set_description(str(key))
            self.new_dim[key] = self.kpca_model(dim, 'linear')
            key = (str(dim) + 'Dim', 'KPCA', 'Polynomial')
            pbar.set_description(str(key))
            self.new_dim[key] = self.kpca_model(dim, 'poly')
            key = (str(dim) + 'Dim', 'KPCA', 'Radial')
            pbar.set_description(str(key))
            self.new_dim[key] = self.kpca_model(dim, 'rbf')
            # No known way of getting the components
            key = (str(dim) + 'Dim', 'LOL', '')
            pbar.set_description(str(key))
            self.new_dim[key] = self.lol_model(dim)
            k = floor(sqrt(min(len(self.X_train), len(self.X_train[0]))))
            key = (str(dim) + 'Dim', 'LPP', 'k=' + str(k))
            pbar.set_description(str(key))
            self.new_dim[key] = self.lpp_model(dim, k)
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

    def slmvp_model(self, n, type_kernel, gammas=None, poly_order=None):
        # Get the principal components
        BAux = SLMVPTrain(X=self.X_train.T, Y=self.y_train,
                          rank=n,
                          typeK=type_kernel,
                          gammaX=gammas,
                          gammaY=gammas,
                          polyValue=poly_order)

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

    def kpca_model(self, n, type_kernel):
        kernel_pca = KernelPCA(
            n_components=n, kernel=type_kernel, fit_inverse_transform=True
        )
        X_kpca_train = kernel_pca.fit(self.X_train).transform(self.X_train)
        X_kpca_test = kernel_pca.transform(self.X_test)

        return X_kpca_train.T, X_kpca_test.T

    def lol_model(self, n):
        lmao = LOL(n_components=n+1, svd_solver='full')
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

    def apply_clf(self):
        """Run classifiers and save new scores in self.scores"""

        print('XGBoost')
        xgb_pipe = Pipeline([('mms', MinMaxScaler()),
                             ('xgb', XGBClassifier())])
        params = [{'xgb__n_estimators': [5, 10, 20, 50, 100]}]
        gs_xgb = GridSearchCV(xgb_pipe,
                              param_grid=params,
                              scoring='accuracy',
                              cv=5)
        pbar = tqdm(self.new_dim.keys())
        for key_dim in pbar:
            pbar.set_description(key_dim)
            gs_xgb.fit(self.new_dim[key_dim][0].T, self.y_train)
            self.scores['XGBoost-'+key_dim] = [
                gs_xgb.score(self.new_dim[key_dim][1].T, self.y_test),
                gs_xgb.best_params_
            ]

        self.pickle_scores(datetime.now().strftime('%m-%d-%H:%M'))

    def get_corr_best_1dim(self):
        # TODO
        pass
