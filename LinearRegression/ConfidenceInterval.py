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