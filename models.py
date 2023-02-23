from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import LocallyLinearEmbedding as LLE
from lol import LOL
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from lpproj import LocalityPreservingProjection
from sklearn.metrics import mean_squared_error, mean_absolute_error
from slmvp import SLMVPTrain, SLMVP_transform
from tqdm import tqdm


class dim_model:
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

        self.new_dim = dict()

    def apply(self, num_dim=[5]):
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

    def lpp_model(self, n, k, X_val='na'):
        lpp = LocalityPreservingProjection(n_components=n, n_neighbors=k)
        lpp.fit(self.X_train)
        X_lpp_train = lpp.transform(self.X_train)
        X_lpp_test = lpp.transform(self.X_test)

        if not isinstance(X_val, str):
            X_lpp_val = lpp.transform(X_val)

        return X_lpp_train, X_lpp_val, X_lpp_test

    def pca_model(self, n, X_val='na'):
        pca_model = PCA(n_components=n).fit(self.X_train)
        X_pca_train = pca_model.transform(self.X_train)
        X_pca_test = pca_model.transform(self.X_test)

        return pca_model, X_pca_train, X_pca_test

    def lle_model(self, neighbors, n, _reg, X_val='na'):
        lle = LLE(n_neighbors=neighbors, n_components=n, reg=_reg)
        X_lle_train = lle.fit_transform(self.X_train)
        X_lle_test = lle.transform(self.X_test)
        if not isinstance(X_val, str):
            X_lle_val = lle.transform(X_val)

        return X_lle_train, X_lle_val, X_lle_test

    def lda_model(self, n, y_train, X_val='na'):
        lda = LDA(n_components=n)
        X_lda_train = lda.fit_transform(self.X_train, y_train)
        X_lda_test = lda.transform(self.X_test)
        if X_val != 'na':
            X_lda_val = lda.transform(X_val)

        return X_lda_train, X_lda_val, X_lda_test

    def kpca_model(self, n, kernel_, X_val='na'):
        kernel_pca = KernelPCA(
            n_components=n, kernel=kernel_, fit_inverse_transform=True
        )
        X_kpca_train = kernel_pca.fit(self.X_train).transform(self.X_train)
        X_kpca_test = kernel_pca.transform(self.X_test)
        if not isinstance(X_val, str):
            X_kpca_val = kernel_pca.transform(X_val)

        return X_kpca_train, X_kpca_val, X_kpca_test

    def lol_model(self, n, y_train, X_val='na'):
        lmao = LOL(n_components=n, svd_solver='full')
        lmao.fit(self.X_train, y_train)
        X_lol_train = lmao.transform(self.X_train)
        X_lol_test = lmao.transform(self.X_test)

        if not isinstance(X_val, str):
            X_lol_val = lmao.transform(X_val)

        return X_lol_train, X_lol_val, X_lol_test


class clf_model:

    def __init__(self):
        pass

    def xgboost_model(self, n_estimators_, X_tr, y_tr, X_val, y_val, X_test, y_test):
        # model = XGBClassifier(n_estimators =n_estimators_ )
        model = XGBRegressor(n_estimators=n_estimators_)
        model.fit(X_tr, y_tr)

        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

        print("################ XGBoost Model ####################")
        print("0.Hyperparameters")
        print("-n_estimators:", n_estimators_)
        print("1.Validation")
        error_squared_val = mean_squared_error(y_val, y_val_pred)
        print("accuracy", error_squared_val)
        print("2.Test")
        error_squared_test = mean_squared_error(y_test, y_test_pred)
        print("accuracy", error_squared_test)

        return [error_squared_val, error_squared_test]
