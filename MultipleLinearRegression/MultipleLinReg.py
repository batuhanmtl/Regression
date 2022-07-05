import numpy as np
import pandas as pd


def Multiple_Regression(data, x_cols_name=None, y_col=''):
    """
    Function that calculates
    the coefficients of the Multiple Linear Regression equation.

    :param data: must be a data frame
    :param x_cols_name: must be a list with column names of non-dependent Variables.
    :param y_col: column name of the dependent variable
    :return: numpy.ndarray -array of beta coefficients in the regression equation-
    """

    if x_cols_name is None:
        x_cols_name = []

    ones = []
    for i in range(len(data)):
        ones.append(1.0)
    ones = np.array(ones)

    ones_X = ones.reshape(len(data), 1)

    X = []

    for i in x_cols_name:
        X.append(data[i].values)

    X = np.array(X)
    X = X.transpose()  # variable values matrix

    for i in range(1, len(x_cols_name) + 1):
        ones_X = np.insert(ones_X, i, data[x_cols_name[i - 1]].values, axis=1)

    X_trans = ones_X.transpose()

    Y = data[y_col].values
    Y = np.array(Y)
    Y = Y.reshape(len(data), 1)

    dot_1 = np.dot(X_trans, ones_X)  # matrix of X'X
    dot_2 = np.dot(X_trans, Y)  # matrix of X'Y

    dot_1_inv = np.linalg.inv(dot_1)  # inverse matrix of X'X

    B = np.dot(dot_1_inv, dot_2)  # beta coefficients matrix

    return B


def Multiple_SSr(data, x_cols_name=None, y_col=''):
    """
    Function that calculates the sum of squares of residuals
    :param data: must be a data frame
    :param x_cols_name: must be a list with column names of non-dependent Variables.
    :param y_col: column name of the dependent variable
    :return: numpy.float SSr -the sum of squares of residuals-
    """
    if x_cols_name is None:
        x_cols_name = []

    ones = []

    for i in range(len(data)):
        ones.append(1.0)

    ones = np.array(ones)

    ones_X = ones.reshape(len(data), 1)

    X = []

    for i in x_cols_name:
        X.append(data[i].values)

    X = np.array(X)
    X = X.transpose()

    for i in range(1, len(x_cols_name) + 1):
        ones_X = np.insert(ones_X, i, data[x_cols_name[i - 1]].values, axis=1)

    X_trans = ones_X.transpose()

    Y = data[y_col].values
    Y = np.array(Y)
    Y = Y.reshape(len(data), 1)

    Y_trans = Y.transpose()

    B_trans = Multiple_Regression(data, x_cols_name=x_cols_name, y_col=y_col).transpose()

    dot_1 = np.dot(Y_trans, Y)
    dot_2 = B_trans.dot(X_trans).dot(Y)

    ssr = dot_1 - dot_2  # Y'Y - B'X'Y

    return ssr[0, 0]


def Variance_Estimator(data, x_cols_name=None, y_col=''):
    """
    Function that estimates the variance using the sum of the squares of the residuals.
    :param data: must be a data frame
    :param x_cols_name: must be a list with column names of non-dependent Variables.
    :param y_col: column name of the dependent variable
    :return:  numpy.float variance value
    """
    ssr = Multiple_SSr(data=data, x_cols_name=x_cols_name, y_col=y_col)

    n = len(data)  # number of observations

    k = len(x_cols_name)  # number of non-dependent variables

    variance_est = ssr / (n - k - 1)  # Variance estimator value

    return variance_est


def Coef_Hypot(data, alpha=0.05, beta=0, variable_name='', x_cols_name=None, y_col=''):
    """
    The function that calculates the hypothesis test according to the desired value of the beta coefficient
    :param alpha: Significance level
    :param variable_name: The name of the variable affected by the coefficient to be tested for the hypothesis
    :param beta: The value of beta at which hypothesis testing is desired
    :param data: must be a data frame
    :param x_cols_name: must be a list with column names of non-dependent Variables.
    :param y_col: column name of the dependent variable
    :return: Test statistic value (float) and T-table value(float)
    """
    from scipy.stats import t
    from math import sqrt

    if x_cols_name is None:
        x_cols_name = []

    ones = []

    for i in range(len(data)):
        ones.append(1.0)

    ones = np.array(ones)

    ones_X = ones.reshape(len(data), 1)

    X = []

    for i in x_cols_name:
        X.append(data[i].values)

    X = np.array(X)

    X = X.transpose()

    for i in range(1, len(x_cols_name) + 1):
        ones_X = np.insert(ones_X, i, data[x_cols_name[i - 1]].values, axis=1)

    X_trans = ones_X.transpose()

    dot_1 = np.dot(X_trans, ones_X)
