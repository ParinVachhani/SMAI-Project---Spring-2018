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
	d_eff = 7 ; iters = 50 ; required_data_mult = 1

	num_vals = 11 ; offset = 40

	d_vals = [100 + offset * i for i in xrange(num_vals)]
	k_vals = [int(2 * required_data_mult * d_eff * np.log(d)) for d in d_vals]
	n_vals = [4 * d for d in d_vals]
	# d_vals = [d_eff for d in d_vals]

	num_cores = multiprocessing.cpu_count()
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

	C_vals = [1.0 for d in d_vals]

	results_list = Parallel(n_jobs=num_cores)(delayed(run_lasso_exp)(d_eff, d_vals[i],
							n_vals[i], k_vals[i], iters, C_vals[i], required_data_mult) for i in xrange(num_vals))

	results = defaultdict(list)

	for elt in results_list:
		for key, val in elt.items():
			results[key].append(val)

	print "----------"
	print "EXPERIMENT COMPLETED."
	print "----------"

	# save final results
	fname = "d_eff={}-iters={}-nvals={}-d0={}-offset={}-final".format(d_eff, iters, num_vals, d_vals[0], offset)
	filename = fname + str(int(time.time()))
	save_obj(results, filename)


def run_lasso_exp(d_eff, d, n, k, iters, C, required_data_mult):

	global_results = defaultdict(list)

	min_beta = 1 ; max_beta = 2
	signal_comp = sorted(np.random.choice(d, int(d_eff), replace=False))
	beta = [random_sign() * np.random.uniform(min_beta, max_beta) if i in signal_comp else 0.0 for i in xrange(d)]

	### define parameters
	params = { "mean_x" : [0 for _ in xrange(d)], "cov_x" : np.eye(d),
	           "mean_e" : 0, "var_e" : 0.1, "formula" : lambda x, y: np.dot(x, y),
	           "beta" : beta, "C" : C, "info_del" : True}

	params["lambda"] = 0.0
	params["metric"] = learning.LassoRegressionMetric
	params["data_dist"] = learning.NonLinearYXDistribution
	params["n"] = n ; params["k"] = k ; params["d"] = d ; params["iters"] = iters

	# regularization parameters
	phi = 2 ; rho = 1 ; gamma = 0.5 ; sigma2 = params["var_e"]

	# print "----------------------------------------------------"
	# print "----------------------------------------------------"
	# print "----------------------------------------------------"
	# print "----------------------------------------------------"
	# print "d = {}, non-zero = {}".format(d, len(signal_comp))
	# print "components = ", sorted(signal_comp)
	# print "values = ", [beta[i] for i in signal_comp]
	# print "----------------------------------------------------"
	# print "n = {}, k = {}, d = {}, d_eff = {}, iters = {}.".format(n, k, d, d_eff, iters)
	# print "k = {} | phase one points = {}".format(k, int(required_data_mult * d_eff * np.log(d)))
	# print "----------------------------------------------------"
	# print "lambda (all obs) = ", lambda_calculator(phi, rho, gamma, sigma2, d, k)
	# print "lambda (only initial budget) = ", lambda_calculator(phi, rho, gamma, sigma2, d, int(required_data_mult * d_eff * np.log(d)))

	print "----------------------------------------------------"
	print "n = {}, k = {}, d = {}, d_eff = {}, iters = {}.".format(n, k, d, d_eff, iters)
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
	rsp.budget_recovery = int(required_data_mult * d_eff * np.log(d))
	rsp.lambda_first_stage = lambda_calculator(phi, rho, gamma, sigma2, d, rsp.budget_recovery)
	rsp.use_ols = True
	rsp.use_all = False

	rsp_all = learning.RandomSparseThresholdChoice("r-sp-thr all OLS", **params)
	rsp_all.budget_recovery = int(required_data_mult * d_eff * np.log(d))
	rsp_all.lambda_first_stage = lambda_calculator(phi, rho, gamma, sigma2, d, rsp.budget_recovery)
	rsp_all.use_ols = True
	rsp_all.use_all = True

	alg_list = [rc, tc, rsp, rsp_all, sc_ols, rs_ols]

	exp = learning.MyExperiment("LassoExp")
	start = time.time()
	print "Started at {}.".format(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(start)))
	# results = exp.run_experiment_n(alg_list, **params)
	results = exp.run_experiment(alg_list, **params)
	end = time.time()
	print "Finished at {}. n = {}, k = {}, d = {}, d_eff = {}, iters = {}.".format(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(end)), n, k, d, d_eff, iters)
	print "Required Time for Computations: {} s.".format(end - start)

	# print "Results: "
	for key, v in results.iteritems():
	#    print key, np.mean(v), np.median(v)
	    global_results[key].append((key, d, n, k, v))

	# save partial results
	fname = "d_eff={}-iters={}-n={}-d={}-k={}-partial".format(d_eff, iters, n, d, k)
	filename = fname + str(run_time)
	save_obj(global_results, filename)

	return global_results



if __name__ == '__main__':
	run_time = int(time.time())
	main()

