import math
from RegressionParameters import *
import numpy as np


def B_hypot(x, y, beta):
    """
    Test Statistics for B
    :param x: List of input levels
    :param y: List of output
    :param beta : Claimed beta value
    :return : Test statistic with t-distribution calculated with the requested beta value
    """
    x = np.array(x)
    x = x.reshape(x.size, 1)

    y = np.array(y)
    y = y.reshape(y.size, 1)

    B = B(x, y)

    sxx = Sxx(x)
    ssr = Ssr(x, y)

    n = x.size

    if beta == 0:
        TS = mt.sqrt((n - 2) * sxx / ssr) * abs(B)

        return TS
    else:
        TS = mt.sqrt((n - 2) * sxx / ssr) * (B - beta)

        return TS


def A_Hypot(x, y, alpha):
    """
    Test Statistics for alpha
    :param x: List of input levels
    :param y: List of output
    :param alpha : Claimed alpha value
    :return : Test statistic with the t-distribution calculated with the desired alpha value
    """

    x = np.array(x)
    x = x.reshape(x.size, 1)

    y = np.array(y)
    y = y.reshape(y.size, 1)

    ssr = Ssr(x, y)

    A = A(x, y)

    n = x.size

    TS = math.sqrt(n * (n - 2) * ssr / (np.sum(x * x) * ssr)) * (A - alpha)

    return TS


def Alpha_BetaX0_Hypot(x, y, x_0):
    """
    Test Statistics for alpha + beta*(X_0)
    :param x: List of input levels
    :param y: List of output levels
    :param x_0: Input level at which hypothesis testing will be done

    :return: A test statistic value with a t-distribution calculated
    with the input level at which the hypothesis will be tested.
    """

    x = np.array(x)
    x = x.reshape(x.size, 1)

    y = np.array(y)
    y = y.reshape(y.size, 1)

    sxx = Sxx(x)

    ssr = Ssr(x, y)

    n = x.size

    TS = math.sqrt((1 / n) + ((x_0 - np.mean(x)) ** 2) / sxx) * math.sqrt(ssr / (x.size - 2))

    return TS
