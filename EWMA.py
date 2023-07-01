import numpy as np
import pandas as pd
import scipy

def EWMA_Vol(l, ret):
    length = len(ret)
    sigma_sq = np.ones(length)
    for i in range(length):
        sigma_sq[i] = ret[0] ** 2 if i == 0 else l * (sigma_sq[i - 1]) + (1 - l) * ret[i - 1] ** 2

    return np.sqrt(sigma_sq)



def EWMA_MLE_obj_func(l, ret):
    length = len(ret)
    sigma_sq = np.ones(length)
    for i in range(length):
        sigma_sq[i] = ret[0] ** 2 if i == 0 else l * (sigma_sq[i - 1]) + (1 - l) * ret[i - 1] ** 2

    rolling_var = ret.rolling(20).var()

    obj = 0

    for i in range(int(length / 2), length):
        obj += (rolling_var[i] - sigma_sq[i]) ** 2

    return obj


def EWMA_lambda_selection(ret):
    def EWMA_MLE_obj(l):
        res = EWMA_MLE_obj_func(l, ret)
        return res

    res = scipy.optimize.minimize_scalar(EWMA_MLE_obj, bounds=(0, 1), method='bounded')

    return res.x
