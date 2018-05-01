from __future__ import division
import learning
import math
import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import scipy
import time
import random
import scipy.special
from collections import defaultdict
from util_sims import *
from joblib import Parallel, delayed
import multiprocessing

def main():
	d_eff = 7 ; iters = 90 ; required_data_mult = 2.25

	num_vals = 11 ; offset = 40

	d_vals = [100 + offset * i for i in xrange(num_vals)]
	k_vals = [int((3/2) * required_data_mult * d_eff * np.log(d)) for d in d_vals]
	n_vals = [4 * d for d in d_vals]

	num_cores = 30 # multiprocessing.cpu_count()
	print "Using {} cores for parallel processing.".format(num_cores)
	print "---------------------------"
	print "d_eff = {}, iters = {}.".format(d_eff, iters)
	print "d_vals = {}".format(d_vals)
	print "n_vals = {}".format(n_vals)
	print "k_vals = {}".format(k_vals)
	print "---------------------------"
	print "d_eff log(d) = ", [int(d_eff * np.log(d)) for d in d_vals]
	print "first stage: C * d_eff * log(d) = ", [int(required_data_mult * d_eff * np.log(d)) for d in d_vals]
	print "second stage: k - C * d_eff * log(d) = ", [k_vals[i] - int(required_data_mult * d_eff * np.log(d)) for i, d in enumerate(d_vals)]

	start_d = time.time()
	print "Started at {}.".format(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start_d)))

	C_vals = [1.0 for d in d_vals]

	iters_job = int(np.ceil(iters / num_cores))

	results_list = Parallel(n_jobs=num_cores)(delayed(run_round_sims)(i, d_eff, d_vals, n_vals, k_vals,
							iters_job, C_vals, required_data_mult) for i in xrange(num_cores))

	results_exp = defaultdict(list)

	for elt in results_list:
		for key, val in elt.items():
			results_exp[key].append(val)

	# round_results[algo] -> ((algo, k, n, d, mse_vals), ...)

	print "----------"
	print "EXPERIMENT COMPLETED."
	print "----------"

	# save final results
	fname = "d_eff={}-iters={}-nvals={}-d0={}-offset={}-final".format(d_eff, iters_job * num_cores, num_vals, d_vals[0], offset)
	filename = fname + str(int(time.time()))
	save_obj(results_exp, filename)

	end_d = time.time()
	print "Required Time for Computations: {} s.".format(end_d - start_d)


def run_round_sims(id_job, d_eff, d_vals, n_vals, k_vals, iters, C_vals, required_data_mult):

	# random.seed(id_job * int(time.time()))
	# np.random.seed(id_job * int(time.time() % 10000))

	sims_results, round_results = [], defaultdict(list)
	num_vals = len(d_vals)

	for i in xrange(num_vals):
		results_r = run_lasso_exp(id_job, d_eff, d_vals[i], n_vals[i], k_vals[i],
					  			iters, C_vals[i], required_data_mult)
		sims_results.append(results_r)

	for r in sims_results:
		for key, val in r.iteritems():
			round_results[key].append(val)
	# round_results[algo] -> ((algo, k, n, d, mse_vals), ...)

	# save partial results
	fname = "d_eff={}-iters={}-nvals={}-d0={}-offset={}-job={}-".format(d_eff, iters, num_vals, d_vals[0], d_vals[1] - d_vals[0], id_job)
	filename = fname + str(time.time())
	save_obj(round_results, filename)

	return round_results


def run_lasso_exp(id_job, d_eff, d, n, k, iters, C, required_data_mult):

	np.random.seed(id_job * d * int(time.time() % 100000))

	global_results = defaultdict(list)

	min_beta = 1 ; max_beta = 2
	signal_comp = sorted(np.random.choice(d, int(d_eff), replace=False))
	beta = [random_sign() * np.random.uniform(min_beta, max_beta) if i in signal_comp else 0.0 for i in xrange(d)]

	### define parameters
	params = { "mean_x" : [0 for _ in xrange(d)], "cov_x" : random_covariance_matrix(d),
	           "mean_e" : 0, "var_e" : 0.1, "formula" : lambda x, y: np.dot(x, y),
	           "beta" : beta, "C" : C, "info_del" : True}

	params["lambda"] = 0.0
	params["metric"] = learning.LassoRegressionMetric
	params["data_dist"] = learning.NonLinearYXDistribution
	params["n"] = n ; params["k"] = k ; params["d"] = d ; params["iters"] = iters

	# regularization parameters
	phi = 2 ; rho = 1 ; gamma = 0.5 ; sigma2 = params["var_e"]

	print "----------------------------------------------------"
	print "(job {}) n = {}, k = {}, d = {}, d_eff = {}, iters = {}.".format(id_job, n, k, d, d_eff, iters)
	print "----------------------------------------------------"

	#### define algorithms

	rc = learning.RandomChoice("random algo", **params)
	rc.lambda_reg = lambda_calculator(phi, rho, gamma, sigma2, d, k)

	tc = learning.ThresholdChoice("thr algo", **params)
	tc.lambda_reg = lambda_calculator(phi, rho, gamma, sigma2, d, k)

	sc_ols = learning.SparseThresholdChoice("sparse-thr OLS right", **params)
	sc_ols.set_sparse_components(signal_comp)
	sc_ols.use_ols = True
	sc_ols.lambda_reg = 0.0

	rs_ols = learning.SparseThresholdChoice("sparse-rs OLS right", **params)
	rs_ols.set_sparse_components(signal_comp)
	rs_ols.use_ols = True
	rs_ols.lambda_reg = 0.0
	rs_ols.random_sampling = True

	rsp = learning.RandomSparseThresholdChoice("r-sp-thr OLS", **params)
	rsp.budget_recovery = int((2/3)*k)
	rsp.lambda_first_stage = lambda_calculator(phi, rho, gamma, sigma2, d, rsp.budget_recovery)
	rsp.use_ols = True
	rsp.use_all = False

	rsp_all = learning.RandomSparseThresholdChoice("r-sp-thr all OLS", **params)
	rsp_all.budget_recovery = int((2/3)*k)
	rsp_all.lambda_first_stage = lambda_calculator(phi, rho, gamma, sigma2, d, rsp.budget_recovery)
	rsp_all.use_ols = True
	rsp_all.use_all = True

	alg_list = [rc, tc, rsp, rsp_all, sc_ols, rs_ols]

	exp = learning.MyExperiment("LassoExp")
	start = time.time()
	print "Started at {}.".format(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start)))
	# results = exp.run_experiment_n(alg_list, **params)
	results_lasso = exp.run_experiment(alg_list, **params)
	end = time.time()
	print "(job {}) Finished at {}. n = {}, k = {}, d = {}, d_eff = {}, iters = {}.".format(id_job, time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(end)), n, k, d, d_eff, iters)
	print "Required Time for Computations: {} s.".format(end - start)

	# print "Results: "
	for key, v in results_lasso.iteritems():
		# print key, np.mean(v), np.median(v)
		global_results[key].append((key, d, n, k, v))

	return global_results


def random_covariance_matrix(d):
    d_weights = [np.random.uniform(0, 3) for _ in range(d)]
    A = np.random.multivariate_normal(np.zeros(d), np.eye(d), 2*d)
    return np.dot(A.T, A) + np.diag(d_weights)


if __name__ == '__main__':
	run_time = int(time.time())
	main()

