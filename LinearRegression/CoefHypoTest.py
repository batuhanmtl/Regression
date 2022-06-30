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