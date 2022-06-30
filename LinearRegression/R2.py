from LinearRegression import RegressionParameters as rp
import numpy as np


def R2(x, y):
    """
    It shows the amount of change in the response variables explained by different input values.
    :param x: List of input levels
    :param y: List of output
    :return: Rate of change in response variables
    """

    x = np.array(x)
    x = x.reshape(x.size, 1)

    y = np.array(y)
    y = y.reshape(y.size, 1)

    syy = rp.Syy(y)
    ssr = rp.Ssr(x, y)

    R_2 = 1 - ssr / syy

    return R_2
