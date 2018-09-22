from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

import math
import numpy as np
import pandas as pd


def parseData():
    X = []
    y = []
    with open('../dat/airfoil_self_noise.dat') as f:
        lines = f.readlines()
        for line in lines:
            entities = map(float, line.strip().split())
            X.append(entities[:5])
            y.append(entities[5:])
    return np.array(X), np.array(y)


def preprocessing(X, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y


def sklearnLinearRegression(X, y):
    lm = linear_model.LinearRegression()
    model = lm.fit(X,y)
    p = model.predict(X)
    return p, math.sqrt(mean_squared_error(p, y))


def myLinearRegression(X, y, alpha=1e-2, maX_iter=10000, eps=1e-20):
    assert X.shape[0] == y.shape[0]
    nrow, ncol = X.shape
    nX = np.hstack((X, np.ones((nrow, 1))))
    A = np.zeros((ncol+1, 1))
    old_rmse = np.inf
    for _ in xrange(maX_iter):
        p = nX.dot(A)
        diff = p - y
        dA = nX.T.dot(diff)
        A = A - alpha / nrow * dA
        p = nX.dot(A)
        rmse = math.sqrt(mean_squared_error(p, y))
        if old_rmse - rmse < eps:
            print 'Converged!'
            break
        old_rmse = rmse
    return p, rmse


myLinearRegression(X, y)


def myNewRegression_calculate(X, A, b):
    sgn_X = (X < 0) * 1.0
    log_X = np.log(np.abs(X))
    mul_X = X * A.T + b.T
    sgn_mul_X = (mul_X < 0) * 1.0
    sum_sgn_mul_X = np.sum(mul_X < 0, axis=1)[np.newaxis].T
    log_mul_X = np.log(np.abs(mul_X))
    sum_log_mul_X = np.sum(log_mul_X, axis=1)[np.newaxis].T
    return sgn_X, log_X, mul_X, sgn_mul_X, sum_sgn_mul_X, log_mul_X, sum_log_mul_X


def myNewRegression_predict(sum_log_mul_X, sum_sgn_mul_X, e):
    return np.exp(sum_log_mul_X) * np.power(-1, sum_sgn_mul_X) + e


def myNewRegression(X, y, alpha=1.25 * 1e-4, maX_iter=10000, eps=1e-20):
    np.random.seed(0)
    assert X.shape[0] == y.shape[0]
    nrow, ncol = X.shape
    A = np.random.normal(0, 1, (ncol, 1))
    b = np.random.normal(0, 1, (ncol, 1))
    e = np.random.normal(0, 1, (1, 1))
    old_rmse = np.inf
    for _ in xrange(maX_iter):
        sgn_X, log_X, mul_X, sgn_mul_X, sum_sgn_mul_X, log_mul_X, sum_log_mul_X = myNewRegression_calculate(X, A, b)
        p = myNewRegression_predict(sum_log_mul_X, sum_sgn_mul_X, e)
        diff = p - y
        dX = np.exp(sum_log_mul_X - log_mul_X + log_X) * np.power(-1, sum_sgn_mul_X - sgn_mul_X + sgn_X)
        dc = np.exp(sum_log_mul_X - log_mul_X) * np.power(-1, sum_sgn_mul_X - sgn_mul_X)
        dA = dX.T.dot(diff)
        db = dc.T.dot(diff)
        de = diff
        A = A - alpha / nrow * dA
        b = b - alpha / nrow * db
        e = e - alpha / nrow * de
        sgn_X, log_X, mul_X, sgn_mul_X, sum_sgn_mul_X, log_mul_X, sum_log_mul_X = myNewRegression_calculate(X, A, b)
        p = myNewRegression_predict(sum_log_mul_X, sum_sgn_mul_X, e)
        rmse = math.sqrt(mean_squared_error(p, y))
        if abs(old_rmse - rmse) < eps:
            print 'Converged!'
            break
        old_rmse = rmse
    return p, rmse

myNewRegression(X, y)
