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