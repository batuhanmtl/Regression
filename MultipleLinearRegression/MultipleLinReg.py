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