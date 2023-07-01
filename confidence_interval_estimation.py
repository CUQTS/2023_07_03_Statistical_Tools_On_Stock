import scipy
import numpy as np
import math

def confidence_interval(ret_ls, est_vol, confidence_level):
    n = len(ret_ls)
    df = n-1
    mean = np.mean(ret_ls)
    std = est_vol/math.sqrt(n)
    left_percentile = scipy.stats.t.ppf((1-confidence_level)/2, df)
    right_percentile = scipy.stats.t.ppf(1-(1 - confidence_level)/2, df)

    left_bound = n*(mean + left_percentile*std) - np.sum(ret_ls[1:])
    right_bound = n*(mean + right_percentile*std) - np.sum(ret_ls[1:])

    return [left_bound, right_bound]