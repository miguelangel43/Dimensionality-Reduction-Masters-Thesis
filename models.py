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
from math import floor, sqrt


class Dim:
    """The output of every classifier is a tuple containing the train and test 
    data projected onto the new dimensions, and the new embeddings

    Key Attributes:
    new_dim - dict. contains a tuple (train, test, embeddings)
    """

    def __init__(self, train, test):
        self.X_train = train[:, :-1]
        self.y_train = train[:, -1]
        self.X_test = test[:, :-1]
        self.y_test = test[:, -1]
        self.new_dim = dict()  # X_train, X_test, components
        self.scores = dict()

    def pickle_dim(self, output_path):
        if len(self.new_dim) == 0:
            print('No dimensions loaded.')
            return
        with open(output_path + '.pkl', 'wb') as f:
            pickle.dump(self.new_dim, f)

    def unpickle_dim(self, input_path):
        with open(input_path, 'rb') as f:
            self.new_dim = pickle.load(f)

    def apply_dim(self, num_dim=[5, 10, 50]):  # 5, 10, 50 dims takes 5min
        """Run dim. red. algorithms and save new features in self.new_dim"""
        pbar = tqdm(num_dim)
        for dim in pbar:
            key = str(dim) + 'Dim-SLMVP-Polynomial-Order=5'
            pbar.set_description(key)
            self.new_dim[key] = self.slmvp_model(
                dim, 'polynomial', poly_order=5)
            key = str(dim) + 'Dim-SLMVP-Linear'
            pbar.set_description(key)
            self.new_dim[key] = self.slmvp_model(dim, 'linear')
            key = str(dim) + 'Dim-SLMVP-Radial-Gammas=0.01'
            pbar.set_description(key)
            self.new_dim[key] = self.slmvp_model(dim, 'radial', gammas=0.01)
            key = str(dim) + 'Dim-SLMVP-Radial-Gammas=0.1'
            pbar.set_description(key)
            self.new_dim[key] = self.slmvp_model(dim, 'radial', gammas=0.1)
            key = str(dim) + 'Dim-SLMVP-Radial-Gammas=1'
            pbar.set_description(key)
            self.new_dim[key] = self.slmvp_model(dim, 'radial', gammas=1)
            key = str(dim) + 'Dim-SLMVP-Radial-Gammas=10'
            pbar.set_description(key)
            self.new_dim[key] = self.slmvp_model(dim, 'radial', gammas=10)
            key = str(dim) + 'Dim-PCA'
            pbar.set_description(key)
            self.new_dim[key] = self.pca_model(dim)
            # No known way of getting the components
            key = str(dim) + 'Dim-KPCA-Linear'
            pbar.set_description(key)
            self.new_dim[key] = self.kpca_model(dim, 'linear')
            key = str(dim) + 'Dim-KPCA-Polynomial'
            pbar.set_description(key)
            self.new_dim[key] = self.kpca_model(dim, 'poly')
            key = str(dim) + 'Dim-KPCA-Radial'
            pbar.set_description(key)
            self.new_dim[key] = self.kpca_model(dim, 'rbf')
            # No known way of getting the components
            key = str(dim) + 'Dim-LOL'
            pbar.set_description(key)
            self.new_dim[key] = self.lol_model(dim)
            k = floor(sqrt(len(self.X_train)))
            key = str(dim) + 'Dim-LPP-k=' + str(k)
            pbar.set_description(key)
            self.new_dim[key] = self.lpp_model(dim, k)
            k = floor(sqrt(len(self.X_train)))
            reg = 0.001
            # No known way of getting the components
            key = str(dim) + 'Dim-LLE-k=' + str(k) + '-reg=' + str(reg)
            pbar.set_description(key)
            self.new_dim[key] = self.lle_model(dim, k, reg)

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
        return X_pca_train.T, X_pca_test.T, pca_model.components_

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
        lmao = LOL(n_components=n, svd_solver='full')
        lmao.fit(self.X_train, self.y_train)
        X_lol_train = lmao.transform(self.X_train)
        X_lol_test = lmao.transform(self.X_test)

        return X_lol_train.T, X_lol_test.T

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


class Clf:

    def __init__(self):
        pass

    def xgboost_model(self, n_estimators_, X_tr, y_tr, X_test, y_test):
        model = XGBClassifier(n_estimators=n_estimators_)
        model.fit(X_tr, y_tr)
        y_test_pred = model.predict(X_test)
        error_squared_test = mean_squared_error(y_test, y_test_pred)

        return error_squared_test

    def svm_model(kernel_, X_tr, y_tr, X_test, y_test):
        svm = SVC(kernel=kernel_)
        model = svm.fit(X_tr, y_tr)
        y_test_pred = model.predict(X_test)

        print("2.Test")
        # accuracy_test = accuracy(y_test,y_test_pred)
        error_squared_test = mean_squared_error(y_test, y_test_pred)
        # error_absolute_test = mean_absolute_error(y_test,y_test_pred)
        # print("mean_squared_error",error_squared_test)
        # print("mean_absolute_error",error_absolute_test)
        print("accuracy", error_squared_test)

        return error_squared_test
