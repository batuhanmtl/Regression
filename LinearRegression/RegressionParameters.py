import numpy as np

def Sxx(x):
    """
    Sum of squares of residualsç
    :param x: List of input levels
    :return: Sxx value
    """
    x = np.array(x)
    x = x.reshape(x.size, 1)

    n = x.size
    sum = 0