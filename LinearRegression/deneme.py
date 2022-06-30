import numpy as np
from LinearRegression.ConfidenceInterval import *
from LinearRegression.R2 import *
from LinearRegression.WeightedLeastSquares import *


leke = [165, 89, 55, 34, 9, 30, 59, 83, 109, 127, 153, 112, 80, 45]
olum = [54.6, 53.3, 56.3, 49.6, 57.1, 45.9, 48.5, 50.1, 52.4, 52.5, 53.2, 51.4, 46, 44.6]

boy = [64, 65, 66, 67, 69, 70, 72, 72, 74, 74, 75, 76]
maas = [91, 94, 88, 103, 77, 96, 105, 88, 122, 102, 90, 114]

baba = [60, 62, 64, 65, 66, 67, 68, 70, 72, 74]
ogul = [63.6, 65.2, 66, 65.5, 66.9, 67.1, 67.4, 68.3, 70.1, 70]

hiz = list([45, 50, 55, 60, 65, 70, 75])
galon = list([24.2, 25, 23.3, 22, 21.5, 20.6, 19.8])

interval = B_Conf_Interval(hiz, galon, 0.05)
print(interval)

interval = Alpha_BetaX0_Conf_Interval(baba, ogul, 68, 0.05)
print(interval)

interval = X0_Conf_Interval(baba, ogul, 68, 0.05)
print(interval)

r2 = R2(baba, ogul)
print(r2)

uzaklik = [0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10]
sure = [15, 15.1, 16.5, 19.9, 27.7, 29.7, 26.7, 35.9, 42, 49.4]

coef = Weighted_Squares(uzaklik, sure)
print(coef)
