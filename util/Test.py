# @Time    : 2023/8/7 16:52
# @Author  : TONGXING


import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


def draw_line():
    # 读取图像
    img_origin = cv.imread('img/img111.jpg', cv.IMREAD_COLOR)
    # 裁剪
    # img = img_origin[200:1000, 2000:]
    # img = img_origin[600:1000, 2000:]
    # 转换为灰度图
    gray = cv.cvtColor(img_origin, cv.COLOR_BGR2GRAY)
    # 高斯滤波
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    # 边缘检测
    # edges = cv.Canny(blur, 100, 200)
    img_bin = cv.threshold(gray, 120, 255, cv.THRESH_BINARY)[1]
    lines = cv.HoughLines(img_bin, 0.8, np.pi / 180, 145)
    print(lines[0])
    # 将检测的线绘制在图像上
    for line in lines:
        rho, theta = line[0]
        print(rho, theta)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # 画点
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        # k = (y2 - y1) / (x2 - x1)
        cv.line(img_origin, (x1, y1), (x2, y2), (0, 255, 0))
    # 显示图像
    plt.subplot(121)
    plt.imshow(img_origin[:, :, ::-1])
    plt.title('img')
    plt.show()





# draw_line()
