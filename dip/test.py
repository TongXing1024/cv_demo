"""
@Time ： 2023/7/17 15:53
@Author ： 佟星
"""
import matplotlib.pyplot as plt
import cv2 as cv
import matplotlib
import numpy as np
from scipy.io import savemat

from util.ImgProcess import ImageProcessC

matplotlib.use('TkAgg')
# 上点(检测)
x_up = [185, 543, 821, 1050, 1198, 1405]
y_up = [512, 552, 566, 576, 571, 609]
# 下点(检测)
x_down = [259, 603, 872, 1076, 1226, 1427]
y_down = [749, 762, 749, 736, 742, 704]
# 存储为矩阵
points = np.row_stack((x_up, y_up, x_down, y_down))
print(points)
savemat('mat/points.mat', {'points': points})


