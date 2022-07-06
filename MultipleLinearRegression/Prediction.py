import numpy as np
import pandas as pd
from scipy.stats import t
from LinearRegression.MultipleLinearRegression import Multiple_SSr, Multiple_Regression
from math import sqrt as sq

def Predict(data, alpha=0.05, x_cols_name=None, y_col='', x_value=None):
    """
    Confidence interval estimator for E[Y|x] when x_ 0 =1
    :param alpha: significance level
    :param x_value: x variable values for interval
    :param data: must be a data frame
    :param x_cols_name: must be a list with column names of non-dependent Variables.
    :param y_col: column name of the dependent variable
    :return: float: lower limit , f;oat:upper limit
    """
    if x_value is None:
        x_value = []

    ones = []

    for i in range(len(data)):
        ones.append(1.0)

    ones = np.array(ones)

    ones_X = ones.reshape(len(data), 1)

    x = [1]

    for i in range(len(x_cols_name)):
        x.append(x_value[i])

    x = np.array(x)

    x = x.reshape(len(x_cols_name) + 1, 1)

    x_trans = x.transpose()