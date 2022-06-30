from scipy.stats import t  # for Values of critical points in T_table
from Regression.LinearRegression import RegressionParameters as rp
from Regression.LinearRegression.CoefHypoTest import *
import math as mt
import numpy as np

def B_Conf_Interval(x, y, alpha):
    """
    Confidence interval estimator for beta
    alpha = significance level
    :param x: List of input levels
    :param y: List of output
    :param alpha: significance level
    :return: Confidence interval for beta
    """

    x = np.array(x)
    x = x.reshape(x.size, 1)

    y = np.array(y)
    y = y.reshape(y.size, 1)

    sxx = rp.Sxx(x)
    ssr = rp.Ssr(x, y)

    n = x.size

    limit_1 = rp.B(x, y) - mt.sqrt(ssr / ((n - 2) * sxx)) * t.ppf(q=alpha / 2, df=n - 2)
    limit_2 = rp.B(x, y) + mt.sqrt(ssr / ((n - 2) * sxx)) * t.ppf(q=alpha / 2, df=n - 2)