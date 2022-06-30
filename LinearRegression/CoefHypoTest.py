import math
from Regression.LinearRegression import RegressionParameters
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

    B = RegressionParameters.B(x, y)

    sxx = RegressionParameters.Sxx(x)
    ssr = RegressionParameters.Ssr(x, y)

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
