import numpy as np
import csv
import pickle
import datetime

def lambda_calculator(phi, rho, gamma, sigma2, d, k):
    term1 = phi * rho / (gamma ** 2)
    term2 = 2 * sigma2 * np.log(d) / k
    return np.sqrt(term1 * term2)

def random_sign():
    if np.random.uniform(0, 1) > 0.5:
        return 1.0
    return -1.0

def unix_to_date(unix):
    return datetime.datetime.fromtimestamp(int(unix)).strftime('%Y-%m-%d %H:%M:%S')

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, 0)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)