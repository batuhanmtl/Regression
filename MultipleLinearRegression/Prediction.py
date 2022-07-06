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

    X = []

    for i in x_cols_name:
        X.append(data[i].values)

    X = np.array(X)

    X = X.transpose()  # variable values matrix

    for i in range(1, len(x_cols_name) + 1):
        ones_X = np.insert(ones_X, i, data[x_cols_name[i - 1]].values, axis=1)

    X_trans = ones_X.transpose()

    dot_1 = np.dot(X_trans, ones_X)  # matrix of X'X

    dot_1_inv = np.linalg.inv(dot_1)  # inverse matrix of X'X

    dot_2 = x_trans.dot(dot_1_inv).dot(x)  # matrix of x'(X'X)^-1x

    n = len(data)

    k = len(x_cols_name)

    ssr = Multiple_SSr(data=data, x_cols_name=x_cols_name, y_col=y_col)

    b = Multiple_Regression(data=data, x_cols_name=x_cols_name, y_col=y_col)

    t_table_value = t.ppf(q=alpha / 2, df=n - k - 1)

    sum_xb = 0

    for i in range(len(x_cols_name)):
        sum_xb += x[i] * b[i, 0]

    limit_1 = sum_xb - sq(ssr / (n - k - 1)) * sq(dot_2) * t_table_value

    limit_2 = sum_xb + sq(ssr / (n - k - 1)) * sq(dot_2) * t_table_value