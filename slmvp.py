import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import polynomial_kernel


def kernelLineal(X, Y):
    K = np.dot(X.T, Y)
    return K  # return the kernel matrix of X and Y


def kernelRBF(X, gammaValue):
    K = rbf_kernel(X.T, gamma=gammaValue)
    return K


def kernelPolynomial(X, Y, polyValue):
    K = polynomial_kernel(X.T, Y.T, degree=polyValue)
    return K


def kernel(X, *args, **kwargs):
    Y = kwargs.get('YValue', None)
    gamma = kwargs.get('gammaValue', None)
    typeK = kwargs.get('typeK', None)
    polynomial = kwargs.get('polyValue', None)

    if (typeK == 'radial'):
        if (gamma != None):
            return kernelRBF(X, gamma)
    if (typeK == 'linear'):
        if (Y is not None):
            return kernelLineal(Y, X)
    if (typeK == 'polynomial'):
        return kernelPolynomial(X, Y, polynomial)
    return None


def SLMVPTrain(X, Y, rank, typeK, gammaX, gammaY, polyValue):  # Parametros typeK, gammas, rank

    # Performs Singular value decomposition
    Ux, sx, Vx = np.linalg.svd(X, full_matrices=False)

    # Creates a matrix of all 0 with shape like
    Sx = np.zeros((sx.shape[0], sx.shape[0]))
    Sx[:sx.shape[0], :sx.shape[0]] = np.diag(sx)

    KXX = kernel(X, typeK=typeK, YValue=X,
                 gammaValue=gammaX, polyValue=polyValue)
    # KXX = kernel(Y, typeK='lineal',YValue = X)

    # Centering KXX
    l = KXX.shape[0]
    j = np.ones(l)

    KXX = KXX - (np.dot(np.dot(j, j.T), KXX))/l - (np.dot(KXX, np.dot(j, j.T))) / \
        l + (np.dot((np.dot(j.T, np.dot(KXX, j))), np.dot(j, j.T)))/(np.power(l, 2))
    Y = np.reshape(Y, (1, Y.size))

    KYY = kernel(Y, typeK=typeK, YValue=Y,
                 gammaValue=gammaY, polyValue=polyValue)
    # KYY = kernel(Y, typeK='lineal',YValue = Y)

    # Centering KYY
    KYY = KYY - (np.dot(np.dot(j, j.T), KYY))/l - (np.dot(KYY, np.dot(j, j.T))) / \
        l + (np.dot((np.dot(j.T, np.dot(KYY, j))), np.dot(j, j.T)))/(np.power(l, 2))

    # Para utilizar en tests
    KXXKYY = np.dot(KXX, KYY)
    KXXKYYR = KXXKYY
    KXXKYYR = np.dot(np.dot(Vx[0:rank, :], KXXKYYR), Vx[0:rank, :].T)

    # Obtaining the linear embedding B
    Ub, Sb, Vb = np.linalg.svd((KXXKYYR), full_matrices=False)
    Sx = Sx[:rank, :rank]
    B = np.dot(np.dot(Ux[:, :rank], np.linalg.inv(Sx)), Ub)

    # Projections on the learned space
    # P = np.dot(B.T,X)
    return B  # return the learned model


def SLMVP_transform(B, X):
    P = np.dot(B, X)
    return P
