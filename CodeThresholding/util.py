from __future__ import division
import learning
import math
import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn
import scipy
import time
import random
import scipy.special
from collections import defaultdict

def generate_normal_data(size, mu=0, sigma=1, dimension=2):
    data, x = [], []
    for d in xrange(dimension):
        x.append(np.array(np.random.normal(mu, sigma, size)))
    data = zip(*x)
    return np.array(data)

def generate_truncated_normal_data_thr(thr, mu=0, sigma=1, d=2, size=1):
    data = []
    while len(data) < size:
        Y = generate_normal_data(1, mu, sigma, d)[0]
        if np.linalg.norm(Y, 2) >= thr:
            data.append(Y)
    return np.array(data)

def generate_truncated_normal_data(n, k, mu=0, sigma=1, d=2, size=-1):
    if size == -1:
        size = k
    data = []
    C = 1.0
    threshold = C * np.sqrt(d + 2 * np.log(n / k))
    while len(data) < size:
        Y = generate_normal_data(1, mu, sigma, d)[0]
        if np.linalg.norm(Y, 2) >= threshold:
            data.append(Y)
    return np.array(data)
    
def incomplete_gamma_func(s, x):
    return scipy.special.gammaincc(s, x) * scipy.special.gamma(s)    
    
def check_diagonally_dominant(M, epsilon=1):
    n, m = M.shape
    for i in xrange(n):
        val = np.abs(M[i,i])
        sum_row = np.sum([np.abs(M[i,j]) for j in xrange(m) if j != i])
        if epsilon * sum_row >= val:
            return False
    return True

def gersgorin_circles(M):
    # columns
    upper, lower = [], []
    n, m = M.shape
    for i in xrange(m):
        r = np.sum([np.abs(M[j, i]) for j in xrange(n) if i != j])
        upper.append(M[i,i] + r)
        lower.append(M[i,i] - r)
    return upper, lower

def my_thr(n, k, d):
    return np.sqrt(d + 2 * np.log(n/k))

def mgf_diag(d, theta, T):
    return 1 - 1/d + (1/(d*((1-2*theta)**(d/2)))) * (incomplete_gamma_func(d/2, (1-2*theta)*(thr ** 2)) / incomplete_gamma_func(d/2, (thr ** 2)/2))

def leverage_values(algo):
    X, Y = algo.return_observations()
    X = np.array(X)
    inv_cov = np.linalg.inv(np.dot(X.T, X))
    hat_matrix = np.dot(np.dot(X, inv_cov), X.T)
    return np.diag(hat_matrix)

def cook_distance(algo):
    X, Y = algo.return_observations()
    b_OLS = np.array(algo.return_OLS())
    p = X.shape[1]
    leverage = list(leverage_values(algo))
    mse = [(Y[i] - np.dot(b_OLS, x)) ** 2 for i, x in enumerate(X)]
    cook = [mse[i] * (lev / ((1 - lev) ** 2)) / (np.mean(mse) * p) for i, lev in enumerate(leverage)]
    return cook

def residuals(algo):
    X, Y = algo.return_observations()
    b_OLS = np.array(algo.return_OLS())
    print "beta: ", b_OLS
    res = [Y[i] - np.dot(b_OLS, x) for i, x in enumerate(X)]
    plt.plot(res)
    plt.title(algo.name)
    plt.show()
    return res

def mean_squared_error(algo):
    res = residuals(algo)
    X, Y = algo.return_observations()
    clf = linear_model.LinearRegression(fit_intercept=False)
    clf.fit(X, Y)
    return np.mean([r ** 2 for r in res]), clf.score(X, Y)

def whitening_transform(cov):
    # cov = V diag(lambdas) V.T
    lambdas, V = np.linalg.eig(cov)
    # U = cov^{-1/2} = D^{-1/2} V.T
    U = np.dot(np.diag([1.0/np.sqrt(i) for i in lambdas]), V.T)
    return U

def whiten_data(data, cov):
    U = whitening_transform(cov)
    white_data = np.dot(U, data.T)
    return white_data.T

def empirical_cov(data):
	d = len(data[0])
	cov = np.zeros((d, d))
	for x in data:
		cov += np.outer(np.array(x), np.array(x))
	return cov / len(data)

def search_C(n, k, d, size=100, mean_x=0, std_x=1):
    
    score_c = []
    candidates_C = np.linspace(0.5, 2, size)
    
    for c in candidates_C:
        thr = c * np.sqrt(d + 2 * np.log(n / k))
        X = generate_normal_data(n, mean_x, std_x, d)
        score_c.append(np.sum([1 for x in X if np.linalg.norm(x) >= thr]) - k)
    
    results = sorted(zip(score_c, candidates_C), key=lambda x: np.abs(x[0]))
    return results[0][1]

def run_linear_regression(X, Y):
	clf = linear_model.LinearRegression(fit_intercept=False)
	clf.fit(X, Y)
	res = clf.predict(X) - Y
	return clf.coef_, res


