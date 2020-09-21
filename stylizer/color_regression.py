""" Regularized Linear Regression with custom regularization function

This is the internal methods for style_prediction.py
The purpose of jonhwa regularization loss is to get the best color value in color RGB domain.
"""
__author__ = "Jonghwa Yim"
__credits__ = ["Jonghwa Yim"]
__copyright__ = "Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved"
__email__ = "jonghwa.yim@samsung.com"

import numpy as np

import scipy.optimize
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def l2_loss(z):
    N = len(z)
    val = 0.5 * np.dot(z, z)
    grad = z
    return val / N, grad / N


def l1_loss(z):
    N = len(z)
    val = np.sum(np.abs(z))
    grad = np.sign(z)
    return val / N, grad / N


def jonhwa_loss(z, MODE_CORRELATION):
    polynomial_degree = 3
    N = len(z)
    if False:
        grad = (np.ones(len(z), dtype=z.dtype) + np.append(np.sign(z[0]),
                                                           np.zeros(N - 1, dtype=z.dtype))) / N
        grad *= (np.sum(z) + np.abs(z[0]))  # chain rule
        return 0.5 * (np.sum(z) + np.abs(z[0])) ** 2 / N, grad
    else:
        # the first term is a second order distance at the point x=1.
        val1 = (np.sum(z) - 1) ** 2
        grad1 = np.ones(len(z), dtype=z.dtype)
        grad1 *= (np.sum(z) - 1)  # chain rule

        # the second term is a second order distance at the point x=0.
        val2 = z[0] ** 2
        grad2 = np.append(np.sign(z[0]), np.zeros(N - 1, dtype=z.dtype))
        grad2 *= np.abs(z[0])  # chain rule

        # the third term is a regularization of the impact of different colors.
        st = 1 + polynomial_degree + 1  # bias term + polynomial degree + first order term
        val3 = 0
        grad3 = np.zeros(grad2.shape, dtype=grad2.dtype)
        if st < len(z):
            jh_factor = len(z[st-1:10])
            val3 = (np.dot(z[st:10], z[st:10]) - z[7]*z[7]) / jh_factor  # np.dot(z-1, z-1)
            grad3 = np.append(np.zeros(st, dtype=z.dtype), z[st:10])
            if MODE_CORRELATION:
                grad3 = np.append(grad3, np.zeros(3, dtype=z.dtype))
            grad3[7] = 0
            grad3 /= jh_factor
        return 0.5 * (val1 + val2 + val3) / N, (grad1 + grad2 + grad3) / N


class FitFailedError(Exception):
    def __init__(self, message, res):
        self.message = message
        self.res = res


class JHColorRegression(BaseEstimator):
    def __init__(self):
        return

    def objective(self, W, X, y, alpha, MODE_CORRELATION):
        # compute training cost/grad
        cost, outer_grad = l2_loss(np.dot(X, W) - y)
        grad = np.dot(outer_grad, X)  # chain rule

        # add regularization cost/grad
        reg_cost, reg_grad = jonhwa_loss(W, MODE_CORRELATION)
        cost += alpha * reg_cost
        grad += alpha * reg_grad

        return cost, grad

    def fit(self, X, y, alpha=1.0, MODE_CORRELATION=True):
        X, y = check_X_y(X, y, y_numeric=True)

        m, n = X.shape
        W = np.zeros(n, dtype=X.dtype)

        res = scipy.optimize.minimize(self.objective, W, args=(X, y, alpha, MODE_CORRELATION), jac=True,
                                      method='L-BFGS-B')

        if res.success:
            self.coef_ = np.float32(res.x)
        else:
            raise FitFailedError("Fit failed: {}".format(res.message), res=res)

        return self


    def fit_var(self, X, X_var, y, alpha=1.0, MODE_CORRELATION=True):
        X, y = check_X_y(X, y, y_numeric=True)

        m, n = X.shape
        W = np.zeros(n, dtype=X.dtype)

        xvar2 = np.linalg.inv(np.sqrt(X_var))
        Xpr = np.matmul(xvar2, X)
        ypr = np.matmul(xvar2, y)
        #Xpr = np.matmul(X_var, X)
        #ypr = np.matmul(X_var, y)

        res = scipy.optimize.minimize(self.objective, W, args=(Xpr, ypr, alpha, MODE_CORRELATION), jac=True,
                                      method='L-BFGS-B')

        if res.success:
            self.coef_ = np.float32(res.x)
        else:
            raise FitFailedError("Fit failed: {}".format(res.message), res=res)

        return self

    def predict(self, X):
        check_is_fitted(self, 'coef_')
        X = check_array(X)

        n_features = self.coef_.shape[0]
        if X.shape[1] != n_features:
            raise ValueError(
                "X must have %d dimensional features, not %d" % (n_features, X.shape[1]))

        y = np.dot(X, self.coef_)

        return y


def test_estimator():
    from sklearn.utils.estimator_checks import check_estimator
    check_estimator(JHColorRegression)


def test_gradients():
    z0 = np.random.randn(9)
    func = lambda z: jonhwa_loss(z)[0]
    grad = lambda z: jonhwa_loss(z)[1]
    ret = scipy.optimize.check_grad(func, grad, z0)
    assert ret < 1e-5


if __name__ == '__main__':
    import nose2

    nose2.main()
