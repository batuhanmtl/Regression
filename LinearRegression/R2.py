from Regression.LinearRegression import RegressionParameters as rp
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