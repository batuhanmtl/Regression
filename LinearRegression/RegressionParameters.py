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

    for i in range(0, n):
        sum += x[i, 0] ** 2
    result = sum - n * np.mean(x) ** 2
    return result

def Syy(y):
    """

    :param y: List of output
    :return: Syy value
    """

    y = np.array(y)
    y = y.reshape(y.size, 1)

    n = y.size
    sum = 0