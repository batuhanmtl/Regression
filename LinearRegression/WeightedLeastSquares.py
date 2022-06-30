from scipy import linalg
import numpy as np

def Weighted_Squares(x, y):
    """
    A function that allows us to obtain the alpha and beta values,
        which are the coefficients of the weighted linear regression equation.
    :param x: List of input levels
    :param y: List of output
    :return: Alpha and beta values, which are the coefficients of the weighted linear regression equation.
    """

    x = np.array(x)
    x = x.reshape(x.size, 1)