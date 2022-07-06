import numpy as np
import pandas as pd
from scipy.stats import t
from LinearRegression.MultipleLinearRegression import Multiple_SSr, Multiple_Regression
from math import sqrt as sq

def Predict(data, alpha=0.05, x_cols_name=None, y_col='', x_value=None):

