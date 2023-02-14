from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import LocallyLinearEmbedding as LLE
from lol import LOL
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from lpproj import LocalityPreservingProjection
from sklearn.metrics import mean_squared_error, mean_absolute_error
from slmvp import SLMVPTrain, SLMVP_transform


class dim_model:

    def __init__(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test

    def slmvp_model(self):
        sett = (3, 'polynomial', None)
        print(sett)
        BAux, Sx = SLMVPTrain(X=X_train.T, Y=y_train,
                              rank=sett[0],
                              typeK=sett[1],
                              gammaX=sett[2],
                              gammaY=sett[2],
                              polyValue=poly_order)

        # Get the principal components
        P_data = SLMVP_transform(BAux.T, X_train.T)

        return P_data

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
