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

    interval = [limit_1, limit_2]

    if limit_1 > limit_2:
        interval[0] = limit_2
        interval[1] = limit_1

    return interval


def A_Conf_Interval(x, y, alpha):
    """
    Confidence interval estimator for A
    :param x: List of input levels
    :param y: List of output
    :param alpha: significance level
    :return: Confidence interval for A
    """
    x = np.array(x)
    x = x.reshape(x.size, 1)

    y = np.array(y)
    y = y.reshape(y.size, 1)

    sxx = rp.Sxx(x)
    ssr = rp.Ssr(x, y)

    n = x.size

    T_table_value = t.ppf(q=alpha / 2, df=n - 2)

    sum = 0

    for i in range(0, n):
        sum += x[i, 0] ** 2

    limit_1 = rp.A(x, y) - mt.sqrt(sum * ssr / (n * (n - 2) * sxx)) * T_table_value
    limit_2 = rp.A(x, y) + mt.sqrt(sum * ssr / (n * (n - 2) * sxx)) * T_table_value

    if limit_1 > limit_2:
        interval[0] = limit_2
        interval[1] = limit_1

    return interval


def Alpha_BetaX0_Conf_Interval(x, y, x0, alpha):
    """
    Confidence interval estimator for A+BX0
    :param x: List of input levels
    :param y: List of output
    :param x0: Confidence interval desired input level
    :param alpha: significance level
    :return: Confidence interval for A+BX0
    """

    x = np.array(x)
    x = x.reshape(x.size, 1)

    y = np.array(y)
    y = y.reshape(y.size, 1)

    n = x.size

    A, B = rp.A(x, y), rp.B(x, y)
    A_BX0 = A + B * x0

    T_table_value = t.ppf(q=alpha / 2, df=n - 2)

    Test_Stat = Alpha_BetaX0_Hypot(x, y, x0)

    limit_1 = A_BX0 - Test_Stat * T_table_value
    limit_2 = A_BX0 + Test_Stat * T_table_value

    interval = [limit_1, limit_2]

    if limit_1 > limit_2:
        interval[0] = limit_2
        interval[1] = limit_1

    return interval

def X0_Conf_Interval(x, y, x0, alpha):
    """
    Prediction range for future response.

    Based on Yi response values corresponding to xi input value Y response prediction interval for
     Xo input level.
    :param x0: Input level
    :param x: List of input levels
    :param y: List of output
    :param alpha:  significance level
    :return: Estimation range for response at xo input level
    """

    x = np.array(x)
    x = x.reshape(x.size, 1)

    y = np.array(y)
    y = y.reshape(y.size, 1)

    sxx = rp.Sxx(x)
    ssr = rp.Ssr(x, y)

    n = x.size

    A, B = rp.A(x, y), rp.B(x, y)

    A_BX0 = A + B * x0

    T_table_value = t.ppf(q=alpha / 2, df=n - 2)

    limit_1 = A_BX0 - mt.sqrt((((n + 1) / n) + (x0 - np.mean(x)) ** 2 / sxx) * (ssr / (n - 2))) * T_table_value
    limit_2 = A_BX0 + mt.sqrt((((n + 1) / n) + (x0 - np.mean(x)) ** 2 / sxx) * (ssr / (n - 2))) * T_table_value
