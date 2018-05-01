from __future__ import division
import math
import numpy as np
import pandas as pd
import random
from sklearn import linear_model
import matplotlib.pyplot as plt
from collections import defaultdict
import csv
from scipy.stats import mstats
from scipy.stats import norm
from scipy.stats import chi2
import scipy

class Distribution(object):

	def __init__(self, name, **params):
		self.name = name
		self.params = params

	def sample(self, size=1):
		pass

class NormalDistribution(Distribution):

	def __init__(self, name, **params):
		super(NormalDistribution, self).__init__(name, **params)
		self.mu = self.params["mean"]
		self.sigma2 = self.params["var"]
		self.dim = 1

	def sample(self, size=1):
		return np.random.normal(self.mu, np.sqrt(self.sigma2), size)

class MultivariateNormalDistribution(Distribution):

	def __init__(self, name, **params):
		super(MultivariateNormalDistribution, self).__init__(name, **params)
		self.mu = self.params["mean"]
		self.cov = self.params["cov"]
		self.dim = len(self.mu)

	def sample(self, size=1):
		return np.random.multivariate_normal(self.mu, self.cov, size)

class LinearYNormalXDistribution(Distribution):

	def __init__(self, name, **params):
		super(LinearYNormalXDistribution, self).__init__(name, **params)
		self.mu = self.params["mean_x"]
		self.cov = self.params["cov_x"]
		self.mu_err = self.params["mean_e"]
		self.var_err = self.params["var_e"]	
		self.beta = self.params["beta"]	
		self.dim = len(self.mu)

	def sample(self, size=1):
		X = np.random.multivariate_normal(self.mu, self.cov, size)
		Y = [np.dot(x.T, self.beta) + np.random.normal(self.mu_err, np.sqrt(self.var_err)) for x in X]
		return X, Y

class NonLinearYXDistribution(Distribution):

    def __init__(self, name, **params):
        super(NonLinearYXDistribution, self).__init__(name, **params)

        ## covariate distribution
        self.mu = self.params["mean_x"]
        self.cov = self.params["cov_x"]
        self.x_dist = np.random.multivariate_normal if "x_dist" not in params else params["x_dist"]

        ## error distribution
        self.mu_err = self.params["mean_e"]
        self.var_err = self.params["var_e"]	
        self.error_dist = np.random.normal if "error_dist" not in params else params["error_dist"]
        
        ## response distribution
        self.beta = self.params["beta"]
        self.psi = 0.0 if "psi" not in params else params["psi"]
        # linear_regression_f = lambda x, y: np.dot(x, y) + self.psi * np.dot(x, x)
        linear_regression_f = lambda x, y: np.dot(x, y) + self.psi * (x[0] ** 2 + x[1] * x[2])
        # linear_regression_f = lambda x, y: np.dot(x, y) + self.psi * y[0] * (x[0] ** 3)
        self.formula = linear_regression_f if "formula" not in params else params["formula"]
        self.dim = len(self.mu)

    def sample(self, size=1):
        Y = []
        X = [self.x_dist(self.mu, self.cov) for i in xrange(size)]
        for i in range(len(X)):
            Y.append(self.formula(X[i], self.beta) + self.error_dist(self.mu_err, np.sqrt(self.var_err)))
        return np.array(X), np.array(Y)

class NonLinearYNormalXDistribution(Distribution):

	def __init__(self, name, **params):
		super(NonLinearYNormalXDistribution, self).__init__(name, **params)

		## covariate distribution
		self.mu = self.params["mean_x"]
		self.cov = self.params["cov_x"]
		
		## error distribution
		self.mu_err = self.params["mean_e"]
		self.var_err = self.params["var_e"]	
		
		## response distribution
		self.beta = self.params["beta"]
		self.degrees = self.params["degrees"]
		self.dim = len(self.mu)

	def sample(self, size=1):
		X = np.random.multivariate_normal(self.mu, self.cov, size)
		Z = np.copy(X)
		for i in range(X.shape[0]):
		    for j in range(X.shape[1]):
		        Z[i, j] = Z[i, j] ** self.degrees[j]
		Y = [self.beta[0] + np.dot(z.T, self.beta[1:]) + np.random.normal(self.mu_err, np.sqrt(self.var_err)) for z in Z]
		return X, Y

class LinearYNonLinearXDistribution(Distribution):

	def __init__(self, name, **params):
		super(LinearYNonLinearXDistribution, self).__init__(name, **params)

		## covariate distribution
		self.mu = self.params["mean_x"]
		self.cov = self.params["cov_x"]
		self.dist = self.params["dist_x"]

		## error distribution
		self.mu_err = self.params["mean_e"]
		self.var_err = self.params["var_e"]	
		
		## response distribution
		self.beta = self.params["beta"]
		self.dim = len(self.mu)

	def compute_inv_cdf(self, q):
	    return self.dist.ppf(q)

	def correlated_marginals_gaussian_copula(self, size, white=False):
	    d = self.cov.shape[0] ; mean = [0 for _ in xrange(d)]
	    data_gaussian = np.random.multivariate_normal(mean, self.cov, size)
	    data = []
	    for x in data_gaussian:
	        obs = [scipy.stats.norm.cdf(elt) for elt in x]
	        obs_marginal = [self.compute_inv_cdf(elt) for elt in obs]
	        data.append(obs_marginal)
	    if white:
	        return self.whiten_data(data)
	    return np.array(data)

	def whitening_transform(self):
	    # cov = V diag(lambdas) V.T
	    lambdas, V = np.linalg.eig(self.cov)
	    # U = cov^{-1/2} = D^{-1/2} V.T
	    U = np.dot(np.diag([1.0/np.sqrt(i) for i in lambdas]), V.T)
	    return U

	def whiten_data(self, data):
	    data = np.array(data)
	    white_data = np.dot(self.whitening_transform(), data.T)
	    return white_data.T

	def weight_calculator(self, size, num_iters, k, n, balance=False, delta=0.05, num_std=2):
	    
	    white = True ; d = self.cov.shape[0]
	    data = self.correlated_marginals_gaussian_copula(5 * size, white)
	    
	    thr = np.percentile([np.linalg.norm(x) for x in data], 100*(1 - k/n))
	    	    
	    weights = [1 for _ in xrange(d)]
	    
	    for i in xrange(num_iters):

	    	if i % 10 == 0:
	    		print "Weights Comp Iter ", i

	        data = self.correlated_marginals_gaussian_copula(size, white)
	        accepted_data = [x for x in data if self.compute_weighted_norm(x, weights) >= thr]
	        
	        means = np.array([0.0 for _ in xrange(d)])
	        for x in accepted_data:
	            means += np.array([elt ** 2 for elt in x])
	        means /= float(d)
	        
	        mean_comp = np.mean(means)
	        std_comp = np.std(means)
	        median_comp = np.median(means)
	        for p in xrange(d):
	            dist_p = means[p] - median_comp
	            weights[p] += - (dist_p / median_comp) * delta
	        
	        if balance:
	        	weights = self.balance_weights_cov(weights, self.cov)
	        else: 
	        	weights = self.normalize_weights(weights)

	        data = self.correlated_marginals_gaussian_copula(size, white)
	        thr = np.percentile([self.compute_weighted_norm(x, weights) for x in data], 100*(1 - k/n))

	    return weights, thr

	def balance_weights_cov(self, weights, cov):
	    
	    groups, groups_def = [], []
	    d = cov.shape[0] ; indep_c = []
	        
	    for i in xrange(d):
	        indep = True
	        for j in xrange(i):
	            if cov[i, j] != 0:
	                indep = False
	                for elt in groups:
	                    if j in elt:
	                        elt.append(i)
	                        break
	                break
	        if indep:
	            groups.append([i])

	    for g in groups:
	    	if len(g) > 1:
	    		groups_def.append(g)
	    	else:
	    		indep_c.append(g[0])

	    groups_def.append(indep_c)

	    new_weights = [w for w in weights]
	    for g in groups_def:
	        sum_w = np.sum([w for i, w in enumerate(weights) if i in g])
	        sum_normalized = sum_w / len(g)
	        for elt in g:
	            new_weights[elt] = sum_normalized

	    return self.normalize_weights(new_weights)

	def compute_weighted_norm(self, datapoint, weights):
	    val = np.sum([weights[i] * (datapoint[i] ** 2) for i in xrange(len(datapoint))])
	    return np.sqrt(val)
	        
	def normalize_weights(self, vals):
	    d = len(vals)
	    return d * (vals / np.sum(vals))

	def sample(self, size=1, white=False):
		X = self.correlated_marginals_gaussian_copula(size, white)
		Y = [np.dot(x.T, self.beta) + np.random.normal(self.mu_err, np.sqrt(self.var_err)) for x in X]
		return X, Y	

class LogisticNormalXDistribution(Distribution):

    def __init__(self, name, **params):
        super(LogisticNormalXDistribution, self).__init__(name, **params)

        ## covariate distribution
        self.mu = self.params["mean_x"]
        self.cov = self.params["cov_x"]
        self.x_dist = np.random.multivariate_normal if "x_dist" not in params else params["x_dist"]
        
        ## response distribution
        self.beta = self.params["beta"]

    def sample(self, size=1):
        Y = []
        X = [self.x_dist(self.mu, self.cov) for i in xrange(size)]
        for i in range(len(X)):
        	p = 1 / (1 + np.exp(- np.dot(X[i], self.beta)))
        	y_val = 1 if np.random.uniform(0, 1) < p else 0
        	Y.append(y_val)
        return np.array(X), np.array(Y)

class Data(object):

	def __init__(self, name, distribution, *args, **kwargs):
		self.name = name
		self.dist = distribution
		self.args = args
		self.kwargs = kwargs

	def sample(self, size=1):
		return self.dist.sample(size)

class RealData(Data):

	def __init__(self, name, dataset, *args, **kwargs):
		self.name = name
		self.data_x, self.data_y = dataset
		self.args = args
		self.kwargs = kwargs

	def sample(self, size=1):
		return self.data_x[:size], self.data_y[:size]

class Algorithm(object):

	def __init__(self, name, *args, **kwargs):
		self.name = name
		self.args = args
		self.kwargs = kwargs
		self.info = []

	def make_decision(self, data, *args, **kwargs):
		pass

	def return_performance(self, *args, **kwargs):
		pass

	def reset(self, *args, **kwargs):
		pass

	def return_observations(self):
		X = [elt[1] for elt in self.info if elt[0] == 1]
		Y = [elt[2] for elt in self.info if elt[0] == 1]
		return np.array(X), np.array(Y)

	def return_all_observations(self):
		X = [elt[1] for elt in self.info]
		Y = [elt[2] for elt in self.info]
		return np.array(X), np.array(Y)

	def return_OLS(self):
		X, Y = self.return_observations()
		clf = linear_model.LinearRegression(fit_intercept=False)
		clf.fit(X, Y)
		return clf.coef_

	def return_residuals(self):
		X, Y = self.return_observations()
		clf = linear_model.LinearRegression(fit_intercept=False)
		clf.fit(X, Y)
		return clf.predict(X) - Y

	def return_W(self, beta):
		W = []
		X, Y = self.return_observations()
		for x in X:
			exp = np.exp(np.dot(x, beta.T))
			W.append(exp/((1 + exp) ** 2))
		return W

	def return_tr_fish_inv(self, beta):
		W = self.return_W(beta)
		X, Y = self.return_observations()
		I = np.dot(X.T, np.dot(np.diag(W), X))
		w, v = np.linalg.eig(I)
		return np.trace(np.linalg.inv(I)), np.min(w), np.max(w), np.sum([1/t for t in w])

	def return_lambdamin_fisherlr(self):
		X, Y = self.return_observations()
		S = np.dot(X.T, X)
		w, v = np.linalg.eig(S)
		return np.min(w), np.max(w)

class RandomChoice(Algorithm):

	def __init__(self, name, *args, **kwargs):
		super(RandomChoice, self).__init__(name, *args, **kwargs)
		self.n = self.kwargs["n"]
		self.k = self.kwargs["k"]

	def make_decision(self, data_x, data_y, *args, **kwargs):
		if len(data_y) <= self.k:
			return 1
		return 0

	def update_info(self, decision, data_x, data_y, *args, **kwargs):
		self.info.append((decision, data_x[-1], data_y[-1]))

	def return_performance(self, *args, **kwargs):
		return self.info

class ThresholdChoice(Algorithm):

	def __init__(self, name, *args, **kwargs):
		super(ThresholdChoice, self).__init__(name, *args, **kwargs)
		self.n = kwargs["n"]
		self.k = kwargs["k"]
		self.d = kwargs["d"]
		self.C = 1 if "C" not in kwargs else kwargs["C"]
		self.threshold = self.compute_threshold()

	def compute_threshold(self):
		return self.C * np.sqrt(self.d + 2 * np.log(self.n / self.k))

	def make_decision(self, data_x, data_y, *args, **kwargs):
		num_seen = len(data_x) - 1
		num_chosen = np.sum([elt[0] for elt in self.info])
		if num_chosen >= self.k:
			return 0
		if self.k - num_chosen == self.n - num_seen:
			return 1
		self.threshold = self.compute_threshold()
		datapoint = data_x[-1]
		decision = 1 if np.linalg.norm(datapoint) >= self.threshold else 0
		return decision

	def update_info(self, decision, data_x, data_y, *args, **kwargs):
		self.info.append((decision, data_x[-1], data_y[-1]))

	def return_performance(self, *args, **kwargs):
		return self.info

class WeightedThresholdChoice(Algorithm):

	def __init__(self, name, *args, **kwargs):
		super(WeightedThresholdChoice, self).__init__(name, *args, **kwargs)
		self.n = kwargs["n"]
		self.k = kwargs["k"]
		self.d = kwargs["d"]
		self.C = 1 if "C" not in kwargs else kwargs["C"]
		self.cov = kwargs["cov_x"]
		self.white_mat = self.whitening_transform(self.cov)
		self.threshold = self.compute_threshold()
		self.apply_opt_weights = True if "apply_weights" not in kwargs else kwargs["apply_weights"]

	def compute_threshold(self):
		return self.C * np.sqrt(self.d + 2 * np.log(self.n / self.k))

	def compute_opt_empirical_threshold(self, weights, dist, n, k, size):
		data, data_y = dist.sample(size)
		wdata = self.whiten_data(data)
		norms = [self.compute_weighted_norm(x, weights) for x in wdata]
		self.opt_thr = np.percentile(norms, 100*(1- k/n))

	def make_decision(self, data_x, data_y, *args, **kwargs):
		num_seen = len(data_x) - 1
		num_chosen = np.sum([elt[0] for elt in self.info])

		if num_chosen >= self.k:
			return 0
		if self.k - num_chosen == self.n - num_seen:
			return 1
		if self.opt_thr:
			thr = self.opt_thr
		else:
			thr = self.compute_threshold()
		if self.apply_opt_weights and len(self.opt_w) > 0:
			weights = self.opt_w
		else:
			weights = [1 for _ in xrange(self.d)]

		datapoint = data_x[-1]
		decision = 1 if self.compute_weighted_norm(self.whiten_data(datapoint), weights) >= thr else 0
		return decision

	def compute_weighted_norm(self, datapoint, weights):
	    val = np.sum([weights[i] * (datapoint[i] ** 2) for i in xrange(len(datapoint))])
	    return np.sqrt(val)

	def whitening_transform(self, cov):
	    # cov = V diag(lambdas) V.T
	    lambdas, V = np.linalg.eig(cov)
	    # U = cov^{-1/2} = D^{-1/2} V.T
	    U = np.dot(np.diag([1.0/np.sqrt(i) for i in lambdas]), V.T)
	    return U

	def whiten_data(self, data):
	    white_data = np.dot(self.white_mat, data.T)
	    return white_data.T

	def update_info(self, decision, data_x, data_y, *args, **kwargs):
		self.info.append((decision, data_x[-1], data_y[-1]))

	def return_performance(self, *args, **kwargs):
		return self.info

class WhiteningThresholdChoice(Algorithm):

	def __init__(self, name, *args, **kwargs):
		super(WhiteningThresholdChoice, self).__init__(name, *args, **kwargs)
		self.n = kwargs["n"]
		self.k = kwargs["k"]
		self.d = kwargs["d"]
		self.cov = kwargs["cov_x"]
		self.white_mat = self.whitening_transform(self.cov)
		self.C = 1 if "C" not in kwargs else kwargs["C"]
		self.threshold = self.compute_threshold()

	def compute_threshold(self):
		return self.C * np.sqrt(self.d + 2 * np.log(self.n / self.k))

	def make_decision(self, data_x, data_y, *args, **kwargs):
		num_chosen = np.sum([elt[0] for elt in self.info])
		if num_chosen >= self.k:
			return 0
		if self.k - num_chosen == self.n - num_seen:
			return 1
		self.threshold = self.compute_threshold()
		datapoint = data_x[-1]
		decision = 1 if np.linalg.norm(self.whiten_data(datapoint)) >= self.threshold else 0
		return decision

	def update_info(self, decision, data_x, data_y, *args, **kwargs):
		self.info.append((decision, data_x[-1], data_y[-1]))

	def return_performance(self, *args, **kwargs):
		return self.info

	def whitening_transform(self, cov):
	    # cov = V diag(lambdas) V.T
	    lambdas, V = np.linalg.eig(cov)
	    # U = cov^{-1/2} = D^{-1/2} V.T
	    U = np.dot(np.diag([1.0/np.sqrt(i) for i in lambdas]), V.T)
	    return U

	def whiten_data(self, data):
	    white_data = np.dot(self.white_mat, data.T)
	    return white_data.T

class SparseThresholdChoice(Algorithm):

	def __init__(self, name, *args, **kwargs):
		super(SparseThresholdChoice, self).__init__(name, *args, **kwargs)
		self.n = kwargs["n"]
		self.k = kwargs["k"]
		self.d = kwargs["d"]
		self.C = 1 if "C" not in kwargs else kwargs["C"]
		self.sparse_dims = range(self.d)
		self.random_sampling = False
		self.threshold = self.compute_threshold()
		self.use_ols = True

	def compute_threshold(self):
		if self.random_sampling:
			return 0.0
		eff_d = len(self.sparse_dims)
		return self.C * np.sqrt(eff_d + 2 * np.log(self.n / self.k))

	def make_decision(self, data_x, data_y, *args, **kwargs):
		num_seen = len(data_x) - 1
		num_chosen = np.sum([elt[0] for elt in self.info])
		if num_chosen >= self.k:
			return 0
		if self.k - num_chosen == self.n - num_seen:
			return 1
		self.threshold = self.compute_threshold()
		datapoint = data_x[-1]
		sparse_norm = np.sqrt(np.sum([datapoint[i] ** 2 for i in self.sparse_dims]))
		decision = 1 if sparse_norm >= self.threshold else 0
		return decision

	def set_sparse_components(self, dimensions):
		self.sparse_dims = dimensions

	def update_info(self, decision, data_x, data_y, *args, **kwargs):
		self.info.append((decision, data_x[-1], data_y[-1]))

	def return_embedded_obs(self, obs, dims):
		return [elt for i, elt in enumerate(obs) if i in dims]

	def return_performance(self, *args, **kwargs):

		# if self.use_ols:
		#	self.lambda_reg = 0.0

		indices_obs = [i for i, elt in enumerate(self.info) if elt[0] == 1][:self.k]
		# in the sparse dims from the first stage
		new_info = []
		for elt in indices_obs:
			if self.use_ols:
				data_p = (1, self.return_embedded_obs(self.info[elt][1], self.sparse_dims), self.info[elt][2])
			else:
				data_p = (1, self.info[elt][1], self.info[elt][2])
			new_info.append(data_p)
		return new_info

class AdaptiveSparseThresholdChoice(Algorithm):

	def __init__(self, name, *args, **kwargs):
		super(AdaptiveSparseThresholdChoice, self).__init__(name, *args, **kwargs)
		self.n = kwargs["n"]
		self.k = kwargs["k"]
		self.d = kwargs["d"]
		self.C = 1 if "C" not in kwargs else kwargs["C"]
		self.sparse_dims = range(self.d)
		self.threshold = self.compute_threshold()
		self.test_time = [int(self.k/2)] if "test" not in kwargs else kwargs["test"]
		self.use_all = True if "use_all" not in kwargs else kwargs["use_all"]
		self.use_ols = False if "use_ols" not in kwargs else kwargs["use_ols"]

	def compute_threshold(self):
		eff_d = len(self.sparse_dims)
		return self.C * np.sqrt(eff_d + 2 * np.log(self.n / self.k))

	def make_decision(self, data_x, data_y, *args, **kwargs):
		num_seen = len(data_x) - 1
		num_chosen = np.sum([elt[0] for elt in self.info])
		if num_chosen >= self.k:
			return 0
		if self.k - num_chosen == self.n - num_seen:
			return 1
		self.threshold = self.compute_threshold()
		datapoint = data_x[-1]
		sparse_norm = np.sqrt(np.sum([datapoint[i] ** 2 for i in self.sparse_dims]))
		decision = 1 if sparse_norm >= self.threshold else 0
		return decision

	def set_sparse_components(self, dimensions):
		self.sparse_dims = dimensions

	def update_info(self, decision, data_x, data_y, *args, **kwargs):
		self.info.append((decision, data_x[-1], data_y[-1]))

		if decision == 1:
			num_chosen = np.sum([elt[0] for elt in self.info])
			if num_chosen in self.test_time:
				# run lasso to change weights
				X = [elt[1] for elt in self.info if elt[0] == 1]
				y = [elt[2] for elt in self.info if elt[0] == 1]
				clf = linear_model.Lasso(alpha=0.01, fit_intercept=False)
				clf.fit(X, y)
				beta_hat = clf.coef_
				self.sparse_dims = [j for j, val in enumerate(beta_hat) if val != 0.0]
				# print self.name, self.sparse_dims
				# print self.sparse_dims, " | we kept ", len(self.sparse_dims), " out of ", len(beta_hat)

	def return_embedded_obs(self, obs, dims):
		return [elt for i, elt in enumerate(obs) if i in dims]

	def return_performance(self, *args, **kwargs):

		if self.use_all:
			num_obs = self.k
		else:
			if self.k >= int(self.test_time[-1]):
				num_obs = self.k - int(self.test_time[-1])
			else:
				num_obs = self.k

		indices_obs = [i for i, elt in enumerate(self.info) if elt[0] == 1]
		# we only choose the observations from the last stage
		indices_obs = indices_obs[-num_obs:]
		# in the sparse dims from the first stage
		new_info = []
		for elt in indices_obs:
			if self.use_ols:
				data_p = (1, self.return_embedded_obs(self.info[elt][1], self.sparse_dims), self.info[elt][2])
			else:
				data_p = (1, self.info[elt][1], self.info[elt][2])
			new_info.append(data_p)
		return new_info

	def reset(self):
		self.sparse_dims = range(self.d)

class AdaptiveSmoothSparseThresholdChoice(Algorithm):

	def __init__(self, name, *args, **kwargs):
		super(AdaptiveSmoothSparseThresholdChoice, self).__init__(name, *args, **kwargs)
		self.n = kwargs["n"]
		self.k = kwargs["k"]
		self.d = kwargs["d"]
		self.C = 1 if "C" not in kwargs else kwargs["C"]
		self.weights = [1.0 for _ in range(self.d)]
		self.threshold = self.compute_threshold()
		self.update_lags = 5 if "update_lags" not in kwargs else kwargs["update_lags"]
		self.delta_w = 0.15 if "delta_w" not in kwargs else kwargs["delta_w"]

	def compute_threshold(self):
		return self.C * np.sqrt(self.d + 2 * np.log(self.n / self.k))

	def compute_weighted_norm(self, datapoint):
		return np.sqrt(np.sum([self.weights[i] * (elt ** 2) for i, elt in enumerate(datapoint)]))

	def make_decision(self, data_x, data_y, *args, **kwargs):
		num_seen = len(data_x) - 1
		num_chosen = np.sum([elt[0] for elt in self.info])
		if num_chosen >= self.k:
			return 0
		if self.k - num_chosen == self.n - num_seen:
			return 1
		self.threshold = self.compute_threshold()
		datapoint = data_x[-1]
		decision = 1 if self.compute_weighted_norm(datapoint) >= self.threshold else 0
		return decision

	def update_info(self, decision, data_x, data_y, *args, **kwargs):
		self.info.append((decision, data_x[-1], data_y[-1]))

		if decision == 1:
			num_chosen = np.sum([elt[0] for elt in self.info])
			if num_chosen % self.update_lags == 0 and num_chosen >= np.log(self.d) ** 2:
				# run lasso to change weights
				X = [elt[1] for elt in self.info if elt[0] == 1]
				y = [elt[2] for elt in self.info if elt[0] == 1]
				clf = linear_model.Lasso(alpha=0.01, fit_intercept=False)
				clf.fit(X, y)
				beta_hat = clf.coef_
				for j, val in enumerate(beta_hat):
					if val != 0.0:
						self.weights[j] += self.delta_w
				# normalize the weigths
				total_w = np.sum(self.weights)
				self.weights = [w/total_w for w in self.weights]
				# print self.name, self.sparse_dims
				# print self.sparse_dims, " | we kept ", len(self.sparse_dims), " out of ", len(beta_hat)

	def return_performance(self, *args, **kwargs):
		return self.info

	def reset(self):
		self.weights = [1.0 for _ in range(self.d)]

class RandomSparseThresholdChoice(Algorithm):

	def __init__(self, name, *args, **kwargs):
		super(RandomSparseThresholdChoice, self).__init__(name, *args, **kwargs)
		self.n = kwargs["n"]
		self.k = kwargs["k"]
		self.d = kwargs["d"]
		self.C = 1 if "C" not in kwargs else kwargs["C"]
		self.sparse_dims = range(self.d)
		self.threshold = self.compute_threshold()
		self.budget_recovery = self.d if "budget_recovery" not in kwargs else kwargs["budget_recovery"] 
		self.lambda_reg = 0.0
		self.lambda_first_stage = 0.01 if "lambda_first" not in kwargs else kwargs["lambda_first"] 
		self.use_all = True if "use_all" not in kwargs else kwargs["use_all"]
		self.use_ols = False if "use_ols" not in kwargs else kwargs["use_ols"]

	def compute_threshold(self):
		eff_d = len(self.sparse_dims)
		return self.C * np.sqrt(eff_d + 2 * np.log(self.n / self.k))

	def make_decision(self, data_x, data_y, *args, **kwargs):
		num_seen = len(data_x) - 1
		num_chosen = np.sum([elt[0] for elt in self.info])
		if num_chosen >= self.k:
			return 0
		if num_seen <= self.budget_recovery:
			return 1
		if self.k - num_chosen == self.n - num_seen:
			return 1
		self.threshold = self.compute_threshold()
		datapoint = data_x[-1]
		sparse_norm = np.sqrt(np.sum([datapoint[i] ** 2 for i in self.sparse_dims]))
		decision = 1 if sparse_norm >= self.threshold else 0
		return decision

	def set_sparse_components(self, dimensions):
		self.sparse_dims = dimensions

	def update_info(self, decision, data_x, data_y, *args, **kwargs):
		self.info.append((decision, data_x[-1], data_y[-1]))

		if decision == 1:
			num_chosen = np.sum([elt[0] for elt in self.info])
			if num_chosen == self.budget_recovery:
				# run lasso to change weights
				X = [elt[1] for elt in self.info if elt[0] == 1]
				y = [elt[2] for elt in self.info if elt[0] == 1]
				clf = linear_model.Lasso(alpha=self.lambda_first_stage, fit_intercept=False)
				clf.fit(X, y)
				beta_hat = clf.coef_
				self.sparse_dims = [j for j, val in enumerate(beta_hat) if val != 0.0]
				# print self.name, self.sparse_dims
				# print self.sparse_dims, " | we kept ", len(self.sparse_dims), " out of ", len(beta_hat)

	def return_embedded_obs(self, obs, dims):
		return [elt for i, elt in enumerate(obs) if i in dims]

	def return_performance(self, *args, **kwargs):
		if self.use_all:
			num_obs = self.k
		else:
			if self.k >= int(self.budget_recovery):
				num_obs = self.k - int(self.budget_recovery)
			else:
				num_obs = self.k

		indices_obs = [i for i, elt in enumerate(self.info) if elt[0] == 1]
		# we only choose the observations from the second stage
		indices_obs = indices_obs[-num_obs:]
		# in the sparse dims from the first stage
		new_info = []
		for elt in indices_obs:
			if self.use_ols:
				data_p = (1, self.return_embedded_obs(self.info[elt][1], self.sparse_dims), self.info[elt][2])
			else:
				data_p = (1, self.info[elt][1], self.info[elt][2])
			new_info.append(data_p)

		return new_info

	def reset(self):
		self.sparse_dims = range(self.d)

class AdaptiveThresholdChoice(Algorithm):

	def __init__(self, name, *args, **kwargs):
		super(AdaptiveThresholdChoice, self).__init__(name, *args, **kwargs)
		self.n = kwargs["n"]
		self.k = kwargs["k"]
		self.d = kwargs["d"]
		self.C = 1 if "C" not in kwargs else kwargs["C"]
		self.threshold = self.compute_threshold(0, 0)

	def compute_threshold(self, num_seen, num_chosen):
		return self.C * np.sqrt(self.d + 2 * np.log((self.n - num_seen + 1) / (self.k - num_chosen + 1)))

	def make_decision(self, data_x, data_y, *args, **kwargs):
		num_seen = len(data_x) - 1
		num_chosen = np.sum([elt[0] for elt in self.info])
		if num_chosen >= self.k:
			return 0
		if self.k - num_chosen == self.n - num_seen:
			return 1
		self.threshold = self.compute_threshold(num_seen, num_chosen)
		datapoint = data_x[-1]
		decision = 1 if np.linalg.norm(datapoint) >= self.threshold else 0
		return decision

	def update_info(self, decision, data_x, data_y, *args, **kwargs):
		self.info.append((decision, data_x[-1], data_y[-1]))

	def return_performance(self, *args, **kwargs):
		return self.info

class WhiteningAdaptiveThresholdChoice(Algorithm):

	def __init__(self, name, *args, **kwargs):
		super(WhiteningAdaptiveThresholdChoice, self).__init__(name, *args, **kwargs)
		self.n = kwargs["n"]
		self.k = kwargs["k"]
		self.d = kwargs["d"]
		self.cov = kwargs["cov_x"]
		self.white_mat = self.whitening_transform(self.cov)
		self.C = 1 if "C" not in kwargs else kwargs["C"]
		self.threshold = self.compute_threshold(0, 0)

	def compute_threshold(self, num_seen, num_chosen):
		return self.C * np.sqrt(self.d + 2 * np.log((self.n - num_seen + 1) / (self.k - num_chosen + 1)))

	def make_decision(self, data_x, data_y, *args, **kwargs):
		num_seen = len(data_x) - 1
		num_chosen = np.sum([elt[0] for elt in self.info])
		if num_chosen >= self.k:
			return 0
		if self.k - num_chosen == self.n - num_seen:
			return 1
		self.threshold = self.compute_threshold(num_seen, num_chosen)
		datapoint = data_x[-1]
		decision = 1 if np.linalg.norm(self.whiten_data(datapoint)) >= self.threshold else 0
		return decision

	def update_info(self, decision, data_x, data_y, *args, **kwargs):
		self.info.append((decision, data_x[-1], data_y[-1]))

	def return_performance(self, *args, **kwargs):
		return self.info

	def whitening_transform(self, cov):
	    # cov = V diag(lambdas) V.T
	    lambdas, V = np.linalg.eig(cov)
	    # U = cov^{-1/2} = D^{-1/2} V.T
	    U = np.dot(np.diag([1.0/np.sqrt(i) for i in lambdas]), V.T)
	    return U

	def whiten_data(self, data):
	    white_data = np.dot(self.white_mat, data.T)
	    return white_data.T

class AdaptiveThresholdUnknownCov(Algorithm):

	def __init__(self, name, *args, **kwargs):
		super(AdaptiveThresholdUnknownCov, self).__init__(name, *args, **kwargs)
		self.n = kwargs["n"]
		self.k = kwargs["k"]
		self.d = kwargs["d"]
		self.C = 1 if "C" not in kwargs else kwargs["C"]
		self.threshold = self.compute_threshold(0, 0)
		self.est_cov = np.zeros((self.d, self.d))

	def compute_threshold(self, num_seen, num_chosen):
		return self.C * np.sqrt(self.d + 2 * np.log((self.n - num_seen + 1) / (self.k - num_chosen + 1)))

	def make_decision(self, data_x, data_y, *args, **kwargs):

		num_seen = len(data_x) - 1
		num_chosen = np.sum([elt[0] for elt in self.info])
		if num_chosen >= self.k:
			return 0
		if self.k - num_chosen == self.n - num_seen:
			return 1

		if len(data_x) == 1:
			self.est_cov = np.zeros((self.d, self.d))
		new_obs_matrix = np.outer(np.array(data_x[-1]), np.array(data_x[-1]))
		self.est_cov += new_obs_matrix

		if num_seen <= self.d * np.log(self.k):
			return 0

		self.threshold = self.compute_threshold(num_seen, num_chosen)
		datapoint = self.whiten_data(np.array(data_x[-1]), len(data_x))
		decision = 1 if np.linalg.norm(datapoint) >= self.threshold else 0
		return decision

	def whitening_transform(self, num_obs):
	    # cov = V diag(lambdas) V.T
	    lambdas, V = np.linalg.eig(self.est_cov / num_obs)
	    # U = cov^{-1/2} = D^{-1/2} V.T
	    U = np.dot(np.diag([1.0/np.sqrt(i) for i in lambdas]), V.T)
	    return U

	def whiten_data(self, datapoint, num_obs):
	    U = self.whitening_transform(num_obs)
	    white_data = np.dot(U, datapoint.T)
	    return white_data.T

	def update_info(self, decision, data_x, data_y, *args, **kwargs):
		self.info.append((decision, data_x[-1], data_y[-1]))

	def return_performance(self, *args, **kwargs):
		return self.info

class PartialAngleLogistic(Algorithm):

	def __init__(self, name, *args, **kwargs):
		super(PartialAngleLogistic, self).__init__(name, *args, **kwargs)
		self.n = kwargs["n"]
		self.k = kwargs["k"]
		self.d = kwargs["d"]
		self.random_sampled = 3 * kwargs["d"] if "random_sampled" not in kwargs else kwargs["random_sampled"]
		self.target_angle = np.pi / 2 if "target_angle" not in kwargs else kwargs["target_angle"]
		self.epsilon = -1 if "epsilon" not in kwargs else kwargs["epsilon"]
		self.partial_beta = []
		self.update_beta = True
		self.batch_size = 1 if "batch_size" not in kwargs else kwargs["batch_size"]

	def make_decision(self, data_x, data_y, *args, **kwargs):
		num_seen = len(data_x) - 1
		num_chosen = np.sum([elt[0] for elt in self.info])
		if num_chosen < self.random_sampled:
			# initially we do random sampling
			return 1
		if num_chosen >= self.k:
			# already done
			return 0
		if self.k - num_chosen == self.n - num_seen:
			# need to choose all remaining
			return 1
		datapoint = data_x[-1]
		angle = self.compute_angle(datapoint)

		if min(np.absolute(angle - self.target_angle), np.absolute(angle - (self.target_angle + np.pi))) <= self.compute_epsilon():
			return 1
		return 0

	def compute_angle(self, point):
		return math.acos(np.dot(point, self.partial_beta.T) / (np.linalg.norm(point) * np.linalg.norm(self.partial_beta)))

	def compute_epsilon(self):
		if self.epsilon != -1:
			return self.epsilon
		return (np.pi/2)*(self.k/self.n)

	def update_info(self, decision, data_x, data_y, *args, **kwargs):
		self.info.append((decision, data_x[-1], data_y[-1]))
		if decision == 1:
			if self.update_beta:
				num_chosen = np.sum([elt[0] for elt in self.info])
				if (num_chosen > self.random_sampled and num_chosen % self.batch_size == 0) or (num_chosen == self.random_sampled):
					# update partial_beta
					X = [elt[1] for elt in self.info if elt[0] == 1][:self.k]
					y = [elt[2] for elt in self.info if elt[0] == 1][:self.k]
					logreg = linear_model.LogisticRegression(C=1e5)
					logreg.fit(X, y)
					self.partial_beta = logreg.coef_[0]

	def return_performance(self, *args, **kwargs):
		return self.info

class BootAngleThrLogistic(Algorithm):

	def __init__(self, name, *args, **kwargs):
		super(BootAngleThrLogistic, self).__init__(name, *args, **kwargs)
		self.n = kwargs["n"]
		self.k = kwargs["k"]
		self.d = kwargs["d"]
		self.C = 1 if "C" not in kwargs else kwargs["C"]
		self.random_sampled = kwargs["k"] / 3.0 if "random_sampled" not in kwargs else kwargs["random_sampled"]
		self.target_angle = np.pi / 2 if "target_angle" not in kwargs else kwargs["target_angle"]
		self.epsilon = -1 if "epsilon" not in kwargs else kwargs["epsilon"]
		self.m_boot = 1 if "m_boot" not in kwargs else kwargs["m_boot"]
		self.partial_beta = []
		self.update_beta = True
		self.factor = 1/2

	def make_decision(self, data_x, data_y, *args, **kwargs):
		num_seen = len(data_x) - 1
		num_chosen = np.sum([elt[0] for elt in self.info])
		if num_chosen < self.k / 3.0:
			# initially we do random sampling
			return 1
		if num_chosen >= self.k:
			# already done
			return 0
		if self.k - num_chosen == self.n - num_seen:
			# need to choose all remaining
			return 1
		datapoint = data_x[-1]

		if np.linalg.norm(datapoint) >= self.compute_threshold():
			num_pass = 0
			eps = self.compute_epsilon()
			for beta_t in self.partial_beta:
				angle = self.compute_angle(datapoint, beta_t)
				if min(np.absolute(angle - self.target_angle), np.absolute(angle - (self.target_angle + np.pi))) <= eps:
						num_pass += 1
			if num_pass >= len(self.partial_beta) / 2:
				return 1

		return 0

	def compute_angle(self, point, beta_val):
		return math.acos(np.dot(point, beta_val.T) / (np.linalg.norm(point) * np.linalg.norm(beta_val)))

	def compute_threshold(self):
		# return self.C * np.sqrt(self.d + 2 * np.log(factor * self.n / self.k))
		return np.sqrt(self.d + self.factor * (1/2) * np.log(self.k))

	def compute_epsilon(self):
		decay = 4
		if self.epsilon != -1:
			return self.epsilon
		return decay * (np.pi/2)*(self.k/self.n)

	def update_info(self, decision, data_x, data_y, *args, **kwargs):
		self.info.append((decision, data_x[-1], data_y[-1]))
		if decision == 1:
			num_chosen = np.sum([elt[0] for elt in self.info])
			if num_chosen >= self.random_sampled and self.update_beta:
				# update partial_beta
				X = [elt[1] for elt in self.info if elt[0] == 1][:self.k]
				y = [elt[2] for elt in self.info if elt[0] == 1][:self.k]
				self.partial_beta = self.bootstrapped_estimates(X, y)

	def bootstrapped_estimates(self, X, Y):
		estimates = []
		m = self.m_boot
		if m == 1:
			# no real bootstrapping
			logreg = linear_model.LogisticRegression(C=1e5)
			logreg.fit(X, Y)
			return [logreg.coef_[0]]

		k_t = len(Y)
		Xa = np.array(X)
		Ya = np.array(Y)
		for i in xrange(m):
			repeat = True
			while repeat:
				indices = np.random.choice(xrange(k_t), k_t, replace=True)
				data_X = [elt for elt in Xa[indices]]
				data_Y = [elt for elt in Ya[indices]]
				if len(np.unique(data_Y)) >= 2:
					repeat = False
			logreg = linear_model.LogisticRegression(C=1e5)
			logreg.fit(data_X, data_Y)
			estimates.append(logreg.coef_[0])

		return estimates


	def return_performance(self, *args, **kwargs):
		return self.info

class BootDotProThrLogistic(Algorithm):

	def __init__(self, name, *args, **kwargs):
		super(BootDotProThrLogistic, self).__init__(name, *args, **kwargs)
		self.n = kwargs["n"]
		self.k = kwargs["k"]
		self.d = kwargs["d"]
		self.C = 1 if "C" not in kwargs else kwargs["C"]
		self.random_sampled = 3 * kwargs["d"] if "random_sampled" not in kwargs else kwargs["random_sampled"]
		self.target_angle = np.pi / 2 if "target_angle" not in kwargs else kwargs["target_angle"]
		self.epsilon = -1 if "epsilon" not in kwargs else kwargs["epsilon"]
		self.m_boot = 1 if "m_boot" not in kwargs else kwargs["m_boot"]
		self.partial_beta = []
		self.update_beta = True

	def make_decision(self, data_x, data_y, *args, **kwargs):
		num_seen = len(data_x) - 1
		num_chosen = np.sum([elt[0] for elt in self.info])
		if num_chosen < self.random_sampled:
			# initially we do random sampling
			return 1
		if num_chosen >= self.k:
			# already done
			return 0
		if self.k - num_chosen == self.n - num_seen:
			# need to choose all remaining
			return 1
		datapoint = data_x[-1]

		if np.linalg.norm(datapoint) >= self.compute_threshold():
			num_pass = 0
			for beta_t in self.partial_beta:
				eps = self.compute_epsilon(beta_t)
				dotpro = np.dot(datapoint, beta_t)
				if np.absolute(dotpro) <= eps:
						num_pass += 1
			if num_pass >= len(self.partial_beta) / 2:
				return 1
		return 0

	def compute_threshold(self):
		return self.C * np.sqrt(self.d + 2 * np.log(self.n / (2 * self.k)))

	def compute_epsilon(self, beta_val):
		if self.epsilon != -1:
			return self.epsilon

		b_norm = np.linalg.norm(beta_val)
		eps = np.sqrt((b_norm ** 2) / 2 * (self.C * np.sqrt(self.d + 2 * np.log(self.n / self.k))))
		return eps

	def update_info(self, decision, data_x, data_y, *args, **kwargs):
		self.info.append((decision, data_x[-1], data_y[-1]))
		if decision == 1:
			num_chosen = np.sum([elt[0] for elt in self.info])
			if num_chosen >= self.random_sampled and self.update_beta:
				# update partial_beta
				X = [elt[1] for elt in self.info if elt[0] == 1][:self.k]
				y = [elt[2] for elt in self.info if elt[0] == 1][:self.k]
				self.partial_beta = self.bootstrapped_estimates(X, y)

	def bootstrapped_estimates(self, X, Y):
		estimates = []
		m = self.m_boot
		if m == 1:
			# no real bootstrapping
			logreg = linear_model.LogisticRegression(C=1e5)
			logreg.fit(X, Y)
			return [logreg.coef_[0]]

		k_t = len(Y)
		Xa = np.array(X)
		Ya = np.array(Y)
		for i in xrange(m):
			indices = np.random.choice(xrange(k_t), k_t, replace=True)
			data_X = [elt for elt in Xa[indices]]
			data_Y = [elt for elt in Ya[indices]]
			logreg = linear_model.LogisticRegression(C=1e5)
			logreg.fit(data_X, data_Y)
			estimates.append(logreg.coef_[0])

		return estimates


	def return_performance(self, *args, **kwargs):
		return self.info

class TwoStepFisherLogistic(Algorithm):

	def __init__(self, name, *args, **kwargs):
		super(TwoStepFisherLogistic, self).__init__(name, *args, **kwargs)
		self.n = kwargs["n"]
		self.k = kwargs["k"]
		self.d = kwargs["d"]
		self.C = 1 if "C" not in kwargs else kwargs["C"]
		self.random_sampled = 3 * kwargs["d"] if "random_sampled" not in kwargs else kwargs["random_sampled"]
		self.target_angle = np.pi / 2 if "target_angle" not in kwargs else kwargs["target_angle"]
		self.epsilon = -1 if "epsilon" not in kwargs else kwargs["epsilon"]
		self.m_boot = 1 if "m_boot" not in kwargs else kwargs["m_boot"]
		self.partial_beta = []
		self.update_beta = True
		self.factor = 0.5 if "factor" not in kwargs else kwargs["factor"]
		self.multiple_updates = True

	def make_decision(self, data_x, data_y, *args, **kwargs):
		num_seen = len(data_x) - 1
		num_chosen = np.sum([elt[0] for elt in self.info])
		if num_chosen < self.random_sampled:
			# initially we do random sampling
			return 1
		if num_chosen >= self.k:
			# already done
			return 0
		if self.k - num_chosen == self.n - num_seen:
			# need to choose all remaining
			return 1

		datapoint = data_x[-1]

		num_pass = 0
		for beta_t in self.partial_beta:
			x_beta, x_orth = self.projection_beta(datapoint, beta_t)
			norm_beta = np.linalg.norm(beta_t)

			c = x_beta[0] / beta_t[0]
			norm_orth = np.linalg.norm(x_orth)

			if np.absolute(c - 2.4 / norm_beta) <= self.compute_epsilon():
				if norm_orth >= self.compute_threshold():
					num_pass += 1
		if num_pass >= len(self.partial_beta) / 2:
			return 1
		
		return 0

	def compute_threshold(self):
		return self.C * np.sqrt(self.d - 1 + 2 * np.log(self.factor * self.n / self.k))
		
	def compute_epsilon(self):
		if self.epsilon != -1:
			return self.epsilon
		return 0.3

	def projection_beta(self, x, beta):
		p_matrix = np.outer(beta.T, beta) / np.dot(beta, beta.T)
		x_beta = np.dot(p_matrix, x)
		x_orth = x - x_beta
		return x_beta, x_orth

	def bootstrapped_estimates(self, X, Y):
		estimates = []
		m = self.m_boot
		if m == 1:
			# no real bootstrapping
			logreg = linear_model.LogisticRegression(C=1e5)
			logreg.fit(X, Y)
			return [logreg.coef_[0]]

		k_t = len(Y)
		Xa = np.array(X)
		Ya = np.array(Y)
		for i in xrange(m):
			indices = np.random.choice(xrange(k_t), k_t, replace=True)
			data_X = [elt for elt in Xa[indices]]
			data_Y = [elt for elt in Ya[indices]]
			logreg = linear_model.LogisticRegression(C=1e5)
			logreg.fit(data_X, data_Y)
			estimates.append(logreg.coef_[0])

		return estimates

	def update_info(self, decision, data_x, data_y, *args, **kwargs):
		self.info.append((decision, data_x[-1], data_y[-1]))
		if decision == 1 and self.update_beta:
			num_chosen = np.sum([elt[0] for elt in self.info])
			if num_chosen == self.random_sampled or (num_chosen > self.random_sampled and self.multiple_updates):
				X = [elt[1] for elt in self.info if elt[0] == 1][:self.k]
				y = [elt[2] for elt in self.info if elt[0] == 1][:self.k]
				self.partial_beta = self.bootstrapped_estimates(X, y)

	def return_performance(self, *args, **kwargs):
		return self.info

class TraceStepChoice(Algorithm):

	def __init__(self, name, *args, **kwargs):
		super(TraceStepChoice, self).__init__(name, *args, **kwargs)
		self.n = kwargs["n"]
		self.k = kwargs["k"]
		self.d = kwargs["d"]
		self.dec = 0.80 if "dec" not in kwargs else kwargs["dec"]

	def compute_threshold(self, data_x, data_y):
		num_chosen = np.sum([elt[0] for elt in self.info])
		num_seen = len(data_x) - 1
		return self.dec + (self.k - num_chosen) / (self.n - num_seen)

	def trace_inv(self, data_x):
	    data_x = np.array(data_x)
	    eigval, eigvec = np.linalg.eig(np.dot(data_x.T, data_x))
	    return np.sum([1.0/eig for eig in eigval])

	def make_decision(self, data_x, data_y, *args, **kwargs):
		chosen_x = [elt[1] for elt in self.info if elt[0] == 1]
		if len(chosen_x) >= self.k:
			return 0
		if len(chosen_x) < self.d:		
			return 1

		prev_trace = self.trace_inv(chosen_x)
		new_trace = self.trace_inv(chosen_x + data_x[-1])

		# print new_trace / prev_trace, self.compute_threshold(data_x, data_y)
		if new_trace / prev_trace <= self.compute_threshold(data_x, data_y):
			# print "Chosen ({})".format(self.compute_threshold(data_x, data_y))
			return 1
		return 0

	def update_info(self, decision, data_x, data_y, *args, **kwargs):
		self.info.append((decision, data_x[-1], data_y[-1]))

	def return_performance(self, *args, **kwargs):
		return self.info

class DimensionBalanced(Algorithm):

	def __init__(self, name, *args, **kwargs):
		super(DimensionBalanced, self).__init__(name, *args, **kwargs)
		self.n = kwargs["n"]
		self.k = kwargs["k"]
		self.d = kwargs["d"]
		self.C = 1 if "C" not in kwargs else kwargs["C"]
		self.threshold = self.compute_threshold(self.d)

	def compute_threshold(self, eff_d):
		return self.C * np.sqrt(eff_d + 2 * np.log(self.n / self.k))

	def make_decision(self, data_x, data_y, *args, **kwargs):
		num_chosen = np.sum([elt[0] for elt in self.info])
		if num_chosen >= self.k:
			return 0

		if num_chosen < self.d:
			self.threshold = self.compute_threshold(self.d)
			datapoint = data_x[-1]
			decision = 1 if np.linalg.norm(datapoint) >= self.threshold else 0
		else:
			# compute max dimension - won't consider it
			chosen_x = [elt[1] for elt in self.info if elt[0] == 1]
			total_info = [np.linalg.norm(np.array(elt)) for elt in zip(*chosen_x)]
			max_dim = sorted(range(len(total_info)), key=lambda x: total_info[x], reverse=True)[0]
			# if the norm of the point in the other dimensions is larger than thr, take it.
			self.threshold = self.compute_threshold(self.d - 1)
			datapoint = [elt for i, elt in enumerate(data_x[-1]) if i != max_dim]
			decision = 1 if np.linalg.norm(datapoint) >= self.threshold else 0
		return decision

	def update_info(self, decision, data_x, data_y, *args, **kwargs):
		self.info.append((decision, data_x[-1], data_y[-1]))

	def return_performance(self, *args, **kwargs):
		return self.info

class RandomlyBalanced(Algorithm):

	def __init__(self, name, *args, **kwargs):
		super(RandomlyBalanced, self).__init__(name, *args, **kwargs)
		self.n = kwargs["n"]
		self.k = kwargs["k"]
		self.d = kwargs["d"]
		self.C = 1 if "C" not in kwargs else kwargs["C"]
		self.threshold = self.compute_threshold(self.d)

	def compute_threshold(self, eff_d):
		return self.C * np.sqrt(eff_d + 2 * np.log(self.n / self.k))

	def make_decision(self, data_x, data_y, *args, **kwargs):
		num_chosen = np.sum([elt[0] for elt in self.info])
		if num_chosen >= self.k:
			return 0

		if num_chosen < self.d:
			self.threshold = self.compute_threshold(self.d)
			datapoint = data_x[-1]
			decision = 1 if np.linalg.norm(datapoint) >= self.threshold else 0
		else:
			# choose random dimension - won't consider it
			max_dim = np.random.randint(self.d)
			# if the norm of the point in the other dimensions is larger than thr, take it.
			self.threshold = self.compute_threshold(self.d - 1)
			datapoint = [elt for i, elt in enumerate(data_x[-1]) if i != max_dim]
			decision = 1 if np.linalg.norm(datapoint) >= self.threshold else 0
		return decision

	def update_info(self, decision, data_x, data_y, *args, **kwargs):
		self.info.append((decision, data_x[-1], data_y[-1]))

	def return_performance(self, *args, **kwargs):
		return self.info

class PartialThresholdChoice(Algorithm):

	def __init__(self, name, gamma=0.8, *args, **kwargs):
		super(PartialThresholdChoice, self).__init__(name, *args, **kwargs)
		self.n = kwargs["n"]
		self.k = kwargs["k"]
		self.d = kwargs["d"]
		self.C = 1 if "C" not in kwargs else kwargs["C"]
		self.threshold = self.compute_threshold()
		self.gamma = gamma

	def compute_threshold(self):
		return self.C * np.sqrt(self.d + 2 * np.log(self.n / self.k))

	def make_decision(self, data_x, data_y, *args, **kwargs):
		num_chosen = np.sum([elt[0] for elt in self.info])
		if num_chosen >= self.k:
			return 0
		if num_chosen < np.ceil((1 - self.gamma) * self.k):
			return 1
		self.threshold = self.compute_threshold()
		datapoint = data_x[-1]
		decision = 1 if np.linalg.norm(datapoint) >= self.threshold else 0
		return decision

	def update_info(self, decision, data_x, data_y, *args, **kwargs):
		self.info.append((decision, data_x[-1], data_y[-1]))

	def return_performance(self, *args, **kwargs):
		return self.info

class CoveringChoice(Algorithm):

	def __init__(self, name, *args, **kwargs):
		super(CoveringChoice, self).__init__(name, *args, **kwargs)
		self.n = kwargs["n"]
		self.k = kwargs["k"]
		self.d = kwargs["d"]
		self.num_planes = int(self.k / 7) if "num_planes" not in kwargs else kwargs["num_planes"]
		self.r_planes = self.random_unit_vectors(self.num_planes, self.d)
		self.num_chi_partitions = int(self.k) + 1 if "num_chi_partitions" not in kwargs else kwargs["num_chi_partitions"]
		self.points_per_partition = 1 if "points_per_partition" not in kwargs else kwargs["points_per_partition"]
		self.set_planes_thresholds()

	def set_planes_thresholds(self):
		chi_dist = chi2(self.d)
		uni_q = [i / self.num_chi_partitions for i in range(1,self.num_chi_partitions)]
		self.thresholds = sorted([chi_dist.ppf(u) for u in uni_q])
		self.balanced_t = defaultdict(list)

	def reset(self, *args, **kwargs):
		return self.set_planes_thresholds()

	def make_decision(self, data_x, data_y, *args, **kwargs):

		decision = 0
		chosen = [elt[1] for elt in self.info if elt[0] == 1]
		
		if len(chosen) >= self.k:
			return 0
		if self.k - len(chosen) == self.n - len(data_y) + 1:
			return 1

		p = data_x[-1]		
		norm_p = np.linalg.norm(p) ** 2

		min_thr = -1
		for elt in self.thresholds:
			if norm_p <= elt:
				min_thr = elt
				break

		if len(self.balanced_t[min_thr]) < self.points_per_partition:
			if self.point_plane_viable(p, chosen, self.r_planes):
				self.balanced_t[min_thr].append(p)
				decision = 1

		return decision

	def update_info(self, decision, data_x, data_y, *args, **kwargs):
		self.info.append((decision, data_x[-1], data_y[-1]))

	def return_performance(self, *args, **kwargs):
		return self.info

	def random_unit_vectors(self, n, d):
	    
	    def random_point(d):
	        x = np.random.normal(0, 1, d)
	        return x / np.linalg.norm(x)
	    
	    return [random_point(d) for i in xrange(n)]

	def choose_points_planes(self, points, planes, k):
	    chosen = []
	    for p in points:
	        if self.point_plane_viable(p, chosen, planes):
	            chosen.append(p)
	            if len(chosen) == k:
	                return chosen
	    return chosen
	    
	def point_plane_viable(self, p, chosen, planes):
	    for q in chosen:
	        sum = 0
	        for v in planes:
	            if np.sign(np.dot(p, v)) == np.sign(np.dot(q, v)):
	                sum += 1
	        if sum == len(planes):
	            return False
	    return True

class TestingChoice(Algorithm):

	def __init__(self, name, *args, **kwargs):
		super(TestingChoice, self).__init__(name, *args, **kwargs)
		self.n = kwargs["n"]
		self.k = kwargs["k"]
		self.d = kwargs["d"]
		self.C = 1 if "C" not in kwargs else kwargs["C"]
		self.threshold = self.compute_threshold(0, 0)
		self.alpha = []
		self.alpha_timestamps = []
		self.last_alpha = -1
		self.decision_pairs = []

	def compute_threshold(self, num_seen, num_chosen):
		return self.C * np.sqrt(self.d + 2 * np.log((self.n - num_seen + 1) / (self.k - num_chosen + 1)))

	def make_decision(self, data_x, data_y, *args, **kwargs):
		
		num_seen = len(data_x) - 1
		num_chosen = np.sum([elt[0] for elt in self.info])

		if num_chosen < self.d:
			return 1
		if num_chosen >= self.k:
			return 0

		if self.last_alpha == -1:
			x_chosen, y_chosen = self.return_observations()
			self.alpha.append(self.compute_alpha(x_chosen, y_chosen))
			self.last_alpha = 0
			self.alpha_timestamps.append(len(data_x))

		alpha_val = self.alpha[-1]

		self.threshold = self.compute_threshold(num_seen, num_chosen)
		decision_th = 1 if np.linalg.norm(data_x[-1]) >= self.threshold else 0

		x_chosen, y_chosen = self.return_observations()
		diff = np.sum(x_chosen, axis=0)
		diff_new = diff + np.array(data_x[-1])
		decision_st = 1 * (np.linalg.norm(diff) > np.linalg.norm(diff_new))

		if decision_th == 1:
			if decision_st == 1:
				decision = 1
			else:
				decision = 1 * (np.random.random() < alpha_val)
		else:
			if decision_st == 0:
				decision = 0
			else:
				decision = 1 * (np.random.random() < 1 - alpha_val)

		if decision == 1:
			self.last_alpha = -1

		self.decision_pairs.append((decision_th, decision_st))

		return decision_th

	def compute_alpha(self, data_x, data_y, m=20, q=75):
		boot_data = self.compute_boot(data_x, data_y, m)
		boot_stat = [self.compute_stat(ds) for ds in boot_data]
		boot_estimate = np.percentile(boot_stat, q)
		return boot_estimate

	def compute_boot(self, data_x, data_y, m=20):
		n = len(data_y)
		choices = [np.random.choice(range(n), size=n, replace=True) for _ in xrange(m)]
		return [(data_x[ind], data_y[ind]) for ind in choices]

	def compute_stat(self, data):
		data_x, data_y = data
		clf = linear_model.LinearRegression(fit_intercept=False)
		clf.fit(data_x, data_y)
		residuals = clf.predict(data_x) - data_y
		statistic, p_val = mstats.normaltest(residuals)
		return p_val

	def update_info(self, decision, data_x, data_y, *args, **kwargs):
		self.info.append((decision, data_x[-1], data_y[-1]))

	def return_performance(self, *args, **kwargs):
		return self.info

class DecisionMaker(object):

	def __init__(self, name, *args, **kwargs):
		self.name = name
		self.args = args
		self.kwargs = kwargs

	def pre_decision(self, algo, data_x, data_y, *args, **kwargs):
		pass

	def make_decision(self, algo, data_x, data_y, *args, **kwargs):
		self.pre_decision(algo, data_x, data_y, *args, **kwargs)
		d = algo.make_decision(data_x, data_y, *args, **kwargs)
		self.post_decision(d, algo, data_x, data_y, *args, **kwargs)

	def post_decision(self, decision, algo, data_x, data_y, *args, **kwargs):
		pass

class ActiveLearningDM(DecisionMaker):

	def __init__(self, name, *args, **kwargs):
		super(ActiveLearningDM, self).__init__(name, *args, **kwargs)

	def post_decision(self, decision, algo, data_x, data_y, *args, **kwargs):
		algo.update_info(decision, data_x, data_y, *args, **kwargs)

class Simulator(object):

	def __init__(self, name, algo_list, data_dist, dm, metric_list, *args, **kwargs):
		self.name = name
		self.algos = algo_list
		self.data = data_dist
		self.dm = dm
		self.metric_list = metric_list
		self.args = args
		self.kwargs = kwargs
		self.results = defaultdict(list)

	def run_simulation(self, *args, **kwargs):

		size_data = kwargs["n"]
		self.sampled_data_x, self.sampled_data_y = self.data.sample(size_data)

		# for m in self.metric_list:
		# 	if type(m) == TestDataMetric:
		# 		sampled_test_x, sampled_test_y = self.data.sample(size_data)
		# 		m.set_testing_data(sampled_test_x, sampled_test_y)

		for alg in self.algos:
			data_alg_x = [elt for elt in self.sampled_data_x]
			data_alg_y = [elt for elt in self.sampled_data_y]
			for i in xrange(size_data):
				self.dm.make_decision(alg, data_alg_x[:i+1], data_alg_y[:i+1], *args, **kwargs)
			for m in self.metric_list:
				self.results[alg.name].append(m.compute(alg))

class Metric(object):
	
	def __init__(self, name, *args, **kwargs):
		self.name = name
		self.args = args
		self.kwargs = kwargs

	def f_performance(self, info):
		pass

	def compute(self, algo):
		return self.f_performance(algo.return_performance())

class CountingMetric(Metric):

	# counts number of chosen observations

	def __init__(self, name, *args, **kwargs):
		super(CountingMetric, self).__init__(name, *args, **kwargs)

	def f_performance(self, info):
		return sum([elt[0] for elt in info])

class LinearRegressionMetric(Metric):

	def __init__(self, name, cov_matrix=[], *args, **kwargs):
		super(LinearRegressionMetric, self).__init__(name, *args, **kwargs)
		self.cov = cov_matrix

	def f_performance(self, info):
		"""
		it regresses y on X, linear model.
		"""
		k = len(info) if "k" not in self.kwargs else self.kwargs["k"]
		X = [elt[1] for elt in info if elt[0] == 1][:k]
		y = [elt[2] for elt in info if elt[0] == 1][:k]
		clf = linear_model.LinearRegression(fit_intercept=False)
		clf.fit(X, y)
		# beta_hat = [clf.intercept_] + list(clf.coef_)
		beta_hat = clf.coef_
		v = np.array(self.kwargs["beta"]) - np.array(beta_hat)
		if len(self.cov) > 0:
			return np.linalg.norm(v) ** 2
		return np.dot(v.T, np.dot(self.cov, v))

class RidgeRegressionMetric(Metric):

	def __init__(self, name, *args, **kwargs):
		super(RidgeRegressionMetric, self).__init__(name, *args, **kwargs)
		self.lambda_val = 0.01 if "lambda" not in kwargs else kwargs["lambda"]

	def f_performance(self, info):
		"""
		it regresses y on X, ridge model.
		"""
		k = len(info) if "k" not in self.kwargs else self.kwargs["k"]
		X = [elt[1] for elt in info if elt[0] == 1][:k]
		y = [elt[2] for elt in info if elt[0] == 1][:k]
		clf = linear_model.Ridge(alpha=self.lambda_val, fit_intercept=False)
		clf.fit(X, y)
		beta_hat = clf.coef_
		return np.linalg.norm(np.array(self.kwargs["beta"]) - np.array(beta_hat)) ** 2

class LassoRegressionMetric(Metric):

	def __init__(self, name, *args, **kwargs):
		super(LassoRegressionMetric, self).__init__(name, *args, **kwargs)
		self.lambda_val = 0.01 if "lambda" not in kwargs else kwargs["lambda"]

	def compute(self, algo):
		return self.f_performance(algo)

	def f_performance(self, algo):
		"""
		it regresses y on X, lasso model.
		"""
		info = algo.return_performance()
		if hasattr(algo, 'lambda_reg'):
			lambda_value = algo.lambda_reg
		else:
			lambda_value = self.lambda_val

		k = len(info) if "k" not in self.kwargs else self.kwargs["k"]
		X = [elt[1] for elt in info if elt[0] == 1][:k]
		y = [elt[2] for elt in info if elt[0] == 1][:k]

		if lambda_value > 0:
			clf = linear_model.Lasso(alpha=lambda_value, fit_intercept=False)
		else:
			clf = linear_model.LinearRegression(fit_intercept=False)
		clf.fit(X, y)
		beta_hat = clf.coef_

		# if we ommited dimensions, need to decompress vector
		if len(beta_hat) < len(self.kwargs["beta"]):
			beta_ext = [0 for elt in self.kwargs["beta"]]
			sparse_dims = algo.sparse_dims
			for i, s in enumerate(sparse_dims):
				beta_ext[s] = beta_hat[i]
			beta_hat = [elt for elt in beta_ext]

		return np.linalg.norm(np.array(self.kwargs["beta"]) - np.array(beta_hat)) ** 2

class LogisticRegressionMetric(Metric):

	def __init__(self, name, *args, **kwargs):
		super(LogisticRegressionMetric, self).__init__(name, *args, **kwargs)

	def f_performance(self, info):
		"""
		it regresses y on X, logistic regression model.
		"""
		k = len(info) if "k" not in self.kwargs else self.kwargs["k"]
		X = [elt[1] for elt in info if elt[0] == 1][:k]
		y = [elt[2] for elt in info if elt[0] == 1][:k]
		logreg = linear_model.LogisticRegression(C=1e5)
		logreg.fit(X, y)
		beta_hat = logreg.coef_
		return np.linalg.norm(np.array(self.kwargs["beta"]) - np.array(beta_hat)) ** 2

class TestDataMetric(Metric):

	def __init__(self, name, *args, **kwargs):
			super(TestDataMetric, self).__init__(name, *args, **kwargs)
			self.test_x = []
			self.test_y = []
	
	def set_testing_data(self, X, Y):
			self.test_x = X
			self.test_y = Y

	def f_performance(self, info):
		"""
		it regresses y on X, and returns test/training mean sq error.
		"""
		k = len(info) if "k" not in self.kwargs else self.kwargs["k"]
		X = [elt[1] for elt in info if elt[0] == 1][:k]
		y = [elt[2] for elt in info if elt[0] == 1][:k]
		clf = linear_model.LinearRegression(fit_intercept=False)
		clf.fit(X, y)
		if len(self.test_x) > 0:
			res = clf.predict(self.test_x) - self.test_y
		else:
			res = clf.predict(X) - y
		return np.mean([r ** 2 for r in res])

class ExternalMetric(Metric):

	def __init__(self, name, f, *args, **kwargs):
			super(ExternalMetric, self).__init__(name, *args, **kwargs)
			self.fun = f

	def f_performance(self, info):
		return self.fun(info)

class Experiment(object):

	def __init__(self, name, *args, **kwargs):
		self.name = name

	def run_experiment(self, algo_list, info, *args, **kwargs):
		pass

class MyExperiment(Experiment):

	def __init__(self, name, *args, **kwargs):
		super(MyExperiment, self).__init__(name, *args, **kwargs)

	def run_experiment(self, algo_list, *args, **kwargs):

		results_experiment = defaultdict(list)

		# read parameters
		n = kwargs["n"] ; k = kwargs["k"] ; d = kwargs["d"]
		num_iters = 100 if "iters" not in kwargs else kwargs["iters"]
		mean_x = np.zeros(d) if "mean_x" not in kwargs else kwargs["mean_x"]
		cov_x = np.eye(d) if "cov_x" not in kwargs else kwargs["cov_x"]
		mean_e = 0 if "mean_e" not in kwargs else kwargs["mean_e"]
		var_e = 0.01 if "var_e" not in kwargs else kwargs["var_e"]
		beta = [np.random.uniform(-5, 5) for j in xrange(d)] if "beta" not in kwargs else kwargs["beta"]
		info_del = False if "info_del" not in kwargs else kwargs["info_del"]
		degrees = [1 for j in xrange(d)] if "degrees" not in kwargs else kwargs["degrees"]
		psi = 0.0 if "non-linear-level" not in kwargs else kwargs["non-linear-level"]

		params_problem = {"n" : n, "k" : k, "d": d, "psi": psi}
		params = {"n" : n, "k" : k, "d": d, "psi": psi,
				  "mean_x" : mean_x, "cov_x" : cov_x,
		          "mean_e" : mean_e, "var_e" : var_e,
		          "beta" : beta, "degrees" : degrees}

		# copy extra parameters
		for elt in kwargs.keys():
			if elt not in params:
				params[elt] = kwargs[elt]

		print params_problem

		data_funct = NonLinearYXDistribution if "data_dist" not in kwargs else kwargs["data_dist"]
		data_dist = data_funct("dataDist", **params)
		data_generator = Data("dataGenerator", data_dist)
		exp_metric = LinearRegressionMetric("LRMetric", **params) if "metric" not in kwargs else kwargs["metric"]("Metric", **params)

		dm = ActiveLearningDM("AL")

		for a in algo_list:
			a.n = n
			a.k = k
			a.reset()

		for i in xrange(num_iters):
			if i % 10 == 0:
				print "Running Iteration {}... k={} | d={} | n={}.".format(i, params_problem["k"],
																		  	  params_problem["d"],
																		      params_problem["n"])

			s = Simulator("mySim", algo_list, data_generator, dm, [exp_metric])
			s.run_simulation(**params_problem)

			for k, v in s.results.iteritems():
				for elt in v:
					results_experiment[k].append(elt)
			for a in algo_list:
				if i == num_iters - 1:
					last_chosen = np.max([q for q, elt in enumerate(a.info) if elt[0] == 1])
					print "{}: {} chosen obs -- last one = {}".format(a.name, np.sum([elt[0] for elt in a.info]), last_chosen)
				if info_del:
					a.info = []
				a.reset()
			del s

		return results_experiment

	def run_experiment_n(self, algo_list, *args, **kwargs):

		results = defaultdict(list)

		n_vals = kwargs["n_vals"]
		k_vals = kwargs["k_vals"]
		psi_vals = [0.0 for elt in n_vals] if "psi_vals" not in kwargs else kwargs["psi_vals"]
		kwargs["info_del"] = True

		for i, n in enumerate(n_vals):
			k = k_vals[i]
			kwargs["n"] = n
			kwargs["k"] = k
			kwargs["non-linear-level"] = psi_vals[i]

			print "-------------------------------"
			print "Computing n={}, k={} ({} iters).".format(n, k, kwargs["iters"])

			r = self.run_experiment(algo_list, **kwargs)
			for key, v in r.iteritems():
				results[key].append((np.mean(v),
									np.median(v),
									np.std(v, ddof=1),
									np.percentile(v, 5, interpolation="midpoint"),
									np.percentile(v, 25, interpolation="midpoint"),
									np.percentile(v, 75, interpolation="midpoint"),
									np.percentile(v, 95, interpolation="midpoint")))

		return results

class MyWeightedExperiment(Experiment):

	def __init__(self, name, *args, **kwargs):
		super(MyWeightedExperiment, self).__init__(name, *args, **kwargs)

	def run_experiment(self, algo_list, *args, **kwargs):

		results_experiment = defaultdict(list)

		# read parameters
		n = kwargs["n"] ; k = kwargs["k"] ; d = kwargs["d"]
		num_iters = 100 if "iters" not in kwargs else kwargs["iters"]
		mean_x = np.zeros(d) if "mean_x" not in kwargs else kwargs["mean_x"]
		cov_x = np.eye(d) if "cov_x" not in kwargs else kwargs["cov_x"]
		mean_e = 0 if "mean_e" not in kwargs else kwargs["mean_e"]
		var_e = 0.01 if "var_e" not in kwargs else kwargs["var_e"]
		beta = [np.random.uniform(-5, 5) for j in xrange(d)] if "beta" not in kwargs else kwargs["beta"]
		info_del = False if "info_del" not in kwargs else kwargs["info_del"]
		degrees = [1 for j in xrange(d)] if "degrees" not in kwargs else kwargs["degrees"]
		psi = 0.0 if "non-linear-level" not in kwargs else kwargs["non-linear-level"]
		nnz_weigths = False if "nnz_weigths" not in kwargs else kwargs["nnz_weigths"]
		nnz_iters = 100 if "nnz_iters" not in kwargs else kwargs["nnz_iters"]
		nnz_size = 3000 if "nnz_size" not in kwargs else kwargs["nnz_size"]
		dist_x = norm if "dist_x" not in kwargs else kwargs["dist_x"]

		params_problem = {"n" : n, "k" : k, "d": d, "psi": psi}
		params = {"n" : n, "k" : k, "d": d, "psi": psi,
				  "mean_x" : mean_x, "cov_x" : cov_x,
		          "mean_e" : mean_e, "var_e" : var_e,
		          "dist_x" : dist_x, "beta" : beta,
		          "degrees" : degrees}

		# copy extra parameters
		for elt in kwargs.keys():
			if elt not in params:
				params[elt] = kwargs[elt]

		print params_problem

		data_funct = NonLinearYXDistribution if "data_dist" not in kwargs else kwargs["data_dist"]
		data_dist = data_funct("dataDist", **params)
		data_generator = Data("dataGenerator", data_dist)
		exp_metric = LinearRegressionMetric("LRMetric", cov_matrix=cov_x, **params) if "metric" not in kwargs else kwargs["metric"]("Metric", **params)

		if nnz_weigths:
			print "Computing Optimal Weights... "
			opt_w, opt_thr = data_dist.weight_calculator(nnz_size, nnz_iters, k, n, delta=0.05, num_std=1)
			print "Optimal Weights: ", opt_w, " | Opt Thr: ", opt_thr, " (iters=", nnz_iters, ", size=", nnz_size, ")"
			for a in algo_list:
				a.opt_w = opt_w
				a.opt_thr = opt_thr
				if hasattr(a, 'apply_opt_weights') and a.apply_opt_weights == False:
					a.compute_opt_empirical_threshold([1 for _ in xrange(d)], data_dist, n, k, 15000)

		dm = ActiveLearningDM("AL")

		for a in algo_list:
			a.n = n
			a.k = k
			a.reset()
			if hasattr(a, 'opt_thr'):
				print "Algo: ", a.name, " | Opt Thr: ", a.opt_thr

		for i in xrange(num_iters):
			if i % 10 == 0:
				print "Running Iteration {}...".format(i)

			s = Simulator("mySim", algo_list, data_generator, dm, [exp_metric])
			s.run_simulation(**params_problem)

			for k, v in s.results.iteritems():
				for elt in v:
					results_experiment[k].append(elt)
			for a in algo_list:
				if i == num_iters - 1:
					chosen_obs = [q for q, elt in enumerate(a.info) if elt[0] == 1]
					last_chosen = np.max(chosen_obs)
					rsampling = 0
					if last_chosen == n-1:
						counter = -1
						while chosen_obs[counter] == n+counter:
							counter -= 1
						rsampling = np.absolute(counter)
					print "{}: {} chosen obs -- last one = {} ({})".format(a.name, np.sum([elt[0] for elt in a.info]), last_chosen, rsampling)
				if info_del:
					a.info = []
				a.reset()
			del s

		return results_experiment

	def run_experiment_n(self, algo_list, *args, **kwargs):

		results = defaultdict(list)

		n_vals = kwargs["n_vals"]
		k_vals = kwargs["k_vals"]
		psi_vals = [0.0 for elt in n_vals] if "psi_vals" not in kwargs else kwargs["psi_vals"]
		kwargs["info_del"] = True

		for i, n in enumerate(n_vals):
			k = k_vals[i]
			kwargs["n"] = n
			kwargs["k"] = k
			kwargs["non-linear-level"] = psi_vals[i]

			print "-------------------------------"
			print "Computing n={}, k={} ({} iters).".format(n, k, kwargs["iters"])

			r = self.run_experiment(algo_list, **kwargs)
			for key, v in r.iteritems():
				print "Algo ", key, " | Mean: ", np.mean(v), " | Median: ", np.median(v)
				results[key].append((np.mean(v),
									np.median(v),
									np.std(v, ddof=1),
									np.percentile(v, 5, interpolation="midpoint"),
									np.percentile(v, 25, interpolation="midpoint"),
									np.percentile(v, 75, interpolation="midpoint"),
									np.percentile(v, 95, interpolation="midpoint")))

		return results

class RealDataExperiment(Experiment):

	def __init__(self, name, data_file, *args, **kwargs):
		super(RealDataExperiment, self).__init__(name, *args, **kwargs)
		self.data_file = data_file
		self.num_obs_summary = defaultdict(list)

	def read_data(self, response=0, std_norm=True, omit_vars=[]):

		data_x, data_y = [], []
		means_x, var_x = [], []
		
		# read data, and standardize the data
		f = open(self.data_file)
		for row in csv.reader(f):
			if response == 0:
				data_y.append(float(row[0]))
				data_x.append([float(num) for num in row[1:]])
			elif response == -1:
				data_y.append(float(row[-1]))
				data_x.append([float(num) for num in row[:-1]])
			else:
				data_y.append(float(row[response]))
				data_x.append([float(num) for i, num in enumerate(row) if i != response])

		# substract means and normalize
		d = len(data_x[0])
		for i in xrange(d):
			vals_i = [elt[i] for elt in data_x]
			means_x.append(np.mean(vals_i))
			var_x.append(np.var(vals_i))
			for elt in data_x:
				elt[i] -= means_x[-1]
				if std_norm:
					if var_x[-1] != 0:
						elt[i] /= np.sqrt(var_x[-1])
					else:
						elt[i] = 0

		self.means_x = means_x if len(omit_vars) == 0 else [val for i, val in enumerate(means_x) if i not in omit_vars]
		self.var_x = var_x if len(omit_vars) == 0 else [val for i, val in enumerate(var_x) if i not in omit_vars]
		self.data_x = data_x if len(omit_vars) == 0 else [[val for i, val in enumerate(elt) if i not in omit_vars] for elt in data_x]

		# substract y-mean
		self.mean_y = np.mean(data_y)
		self.data_y = [y - self.mean_y for y in data_y]

	def sample_data(self, n):
		training_indices = random.sample(range(len(self.data_x)), n)
		test_indices = [i for i in range(len(self.data_x)) if i not in training_indices]
		tr_x = [self.data_x[i] for i in training_indices]
		tr_y = [self.data_y[i] for i in training_indices]
		te_x = [self.data_x[i] for i in test_indices]
		te_y = [self.data_y[i] for i in test_indices]

		return tr_x, tr_y, te_x, te_y

	def run_experiment(self, algo_list, *args, **kwargs):

		results_experiment = defaultdict(list)

		# read parameters
		n = kwargs["n"] ; k = kwargs["k"] ; d = kwargs["d"]
		num_iters = 1 if "iters" not in kwargs else kwargs["iters"]
		info_del = False if "info_del" not in kwargs else kwargs["info_del"]
		rsampling = 0 ; params_problem = {"n" : n, "k" : k, "d": d}

		# extra parameters
		for elt in kwargs.keys():
			if elt not in params_problem:
				params_problem[elt] = kwargs[elt]

		print params_problem

		for a in algo_list:
			a.n = n
			a.k = k
			a.reset()

		for i in xrange(num_iters):
			if i % 10 == 0:
				print "Running Iteration {}...".format(i)

			dm = ActiveLearningDM("AL")
			tr_x, tr_y, te_x, te_y = self.sample_data(n)

			data_generator = RealData("realDataTraining", [tr_x, tr_y])
			test_metric = TestDataMetric("TestMetric", **params_problem)
			test_metric.set_testing_data(te_x, te_y)

			s = Simulator("mySim", algo_list, data_generator, dm, [test_metric])
			s.run_simulation(**params_problem)

			for k, v in s.results.iteritems():
				for elt in v:
					results_experiment[k].append(elt)
			for a in algo_list:
				if i == num_iters - 1:
					last_chosen = np.max([q for q, elt in enumerate(a.info) if elt[0] == 1])
					print "{}: {} chosen obs -- last one = {} ({} rs)".format(a.name, np.sum([elt[0] for elt in a.info]), last_chosen, rsampling)
				self.num_obs_summary[a.name].append((len(a.info), np.sum([elt[0] for elt in a.info])))
				if info_del:
					a.info = []
				a.reset()
			del s

		return results_experiment

	def run_experiment_n(self, algo_list, *args, **kwargs):

		results = defaultdict(list)

		n_vals = kwargs["n_vals"]
		k_vals = kwargs["k_vals"]
		kwargs["info_del"] = True

		for i, n in enumerate(n_vals):
			k = k_vals[i]
			kwargs["n"] = n
			kwargs["k"] = k

			print "-------------------------------"
			print "Computing n={}, k={} ({} iters).".format(n, k, kwargs["iters"])

			r = self.run_experiment(algo_list, **kwargs)
			for key, v in r.iteritems():
				results[key].append((np.mean(v),
									np.median(v),
									np.std(v, ddof=1),
									np.percentile(v, 5, interpolation="midpoint"),
									np.percentile(v, 25, interpolation="midpoint"),
									np.percentile(v, 75, interpolation="midpoint"),
									np.percentile(v, 95, interpolation="midpoint")))
		return results
	
class LogisticExperiment(Experiment):

	def __init__(self, name, *args, **kwargs):
		super(LogisticExperiment, self).__init__(name, *args, **kwargs)
		self.beta_record = []

	def run_experiment(self, algo_list, *args, **kwargs):

		results_experiment = defaultdict(list)

		# read parameters
		n = kwargs["n"] ; k = kwargs["k"] ; d = kwargs["d"]
		num_iters = 100 if "iters" not in kwargs else kwargs["iters"]
		mean_x = np.zeros(d) if "mean_x" not in kwargs else kwargs["mean_x"]
		cov_x = np.eye(d) if "cov_x" not in kwargs else kwargs["cov_x"]
		beta = [np.random.uniform(-5, 5) for j in xrange(d)] if "beta" not in kwargs else kwargs["beta"]
		self.beta_record.append(beta)
		info_del = False if "info_del" not in kwargs else kwargs["info_del"]

		params_problem = {"n" : n, "k" : k, "d": d}
		params = {"n" : n, "k" : k, "d": d, "beta": beta,
				  "mean_x" : mean_x, "cov_x" : cov_x}

		# copy extra parameters
		for elt in kwargs.keys():
			if elt not in params:
				params[elt] = kwargs[elt]

		print params_problem

		dm = ActiveLearningDM("AL")

		for a in algo_list:
			a.n = n
			a.k = k
			a.reset()

		for i in xrange(num_iters):
			if i % 10 == 0:
				print "Running Iteration {}...".format(i)

			beta = [np.random.uniform(-5, 5) for j in xrange(d)] if "beta" not in kwargs else kwargs["beta"]
			params["beta"] = beta

			data_funct = LogisticNormalXDistribution if "data_dist" not in kwargs else kwargs["data_dist"]
			data_dist = data_funct("dataDist", **params)
			data_generator = Data("dataGenerator", data_dist)
			exp_metric = LogisticRegressionMetric("LogRMetric", **params) if "metric" not in kwargs else kwargs["metric"]("Metric", **params)

			s = Simulator("mySim", algo_list, data_generator, dm, [exp_metric])
			s.run_simulation(**params_problem)

			for k, v in s.results.iteritems():
				for elt in v:
					results_experiment[k].append(elt)
			for a in algo_list:
				if i == num_iters - 1:
					chosen_obs = [q for q, elt in enumerate(a.info) if elt[0] == 1]
					last_chosen = np.max(chosen_obs)
					rsampling = 0
					if last_chosen == n-1:
						counter = -1
						while chosen_obs[counter] == n+counter:
							counter -= 1
						rsampling = np.absolute(counter)

					print "{}: {} chosen obs -- last one = {} ({} rs)".format(a.name, np.sum([elt[0] for elt in a.info]), last_chosen, rsampling)
				if info_del:
					a.info = []
				a.reset()
			del s

		return results_experiment

	def run_experiment_n(self, algo_list, *args, **kwargs):

		results = defaultdict(list)

		n_vals = kwargs["n_vals"]
		k_vals = kwargs["k_vals"]
		kwargs["info_del"] = True

		for i, n in enumerate(n_vals):
			k = k_vals[i]
			kwargs["n"] = n
			kwargs["k"] = k

			print "-------------------------------"
			print "Computing n={}, k={} ({} iters).".format(n, k, kwargs["iters"])

			r = self.run_experiment(algo_list, **kwargs)
			for key, v in r.iteritems():
				results[key].append((np.mean(v),
									np.median(v),
									np.std(v, ddof=1),
									np.percentile(v, 5, interpolation="midpoint"),
									np.percentile(v, 25, interpolation="midpoint"),
									np.percentile(v, 75, interpolation="midpoint"),
									np.percentile(v, 95, interpolation="midpoint")))

		return results



