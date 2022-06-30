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

    for i in range(0, n):
        sum += y[i, 0] ** 2
    result = sum - n * np.mean(y) ** 2

    return result


def Sxy(x, y):
    """

    :param x: List of input levels
    :param y: List of output
    :return: Sxy value
    """
    x = np.array(x)
    x = x.reshape(x.size, 1)

    y = np.array(y)
    y = y.reshape(y.size, 1)

    n = x.size
    sum = 0

    for i in range(0, n):
        sum += x[i, 0] * y[i, 0]
    result = sum - n * np.mean(x) * np.mean(y)

    return result


def Ssr(x, y):
    """

    :param x: List of input levels
    :param y: List of output:
    :return: SSr Value
    """

    sxx = Sxx(x)
    syy = Syy(y)
    sxy = Sxy(x, y)

    result = (sxx * syy - sxy ** 2) / sxx

    return result

def B(x, y):
    """
    Estimator of beta in alpha + beta*x regression equation

    :param x: List of input levels
    :param y: List of output
    :return: The beta value in the alpha + beta*x regression equation
    """