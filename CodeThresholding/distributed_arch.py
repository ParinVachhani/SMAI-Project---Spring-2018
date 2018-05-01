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

	num_cores = multiprocessing.cpu_count()
	print "Using {} cores for parallel processing.".format(num_cores)

	results_list = Parallel(n_jobs=num_cores)(delayed(distributed_fun)(i) for i in xrange(num_cores))

	print results_list


def distributed_fun(id_job):

	# random.seed(id_job * int(time.time()))
	np.random.seed(id_job * int(time.time() % 10000))

	rnum = np.random.random()

	print "(Job {}) Random Number = {}.".format(id_job, rnum)

	return rnum




if __name__ == '__main__':
	run_time = int(time.time())
	main()

