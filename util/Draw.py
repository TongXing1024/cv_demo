# @Time    : 2023/8/11 16:38
# @Author  : TONGXING
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

matplotlib.use('TkAgg')


class DrawC:
    """
    画线
    鼠标回调函数
    """

    def draw_line(self, points, lint_parameters):
        """
        绘制原始数据点和拟合的直线,此方法第一个参数传入处理好的坐标，第二个参数传入拟合的直线的参数
        :param points: 原始数据点 shape:(n,2)，n为点的个数，每个点的坐标为(x,y)
        :param lint_parameters: 直线的参数，斜率和截距 shape:(1,2)，第一个元素为斜率，第二个元素为截距
        :return:
        """
        # 绘制原始数据点
        plt.scatter(points[:, 0], points[:, 1], label='datapoint')  #
        # 绘制拟合的直线
        x = np.linspace(0, 10, 100)
        y = np.polyval(lint_parameters, x)
        plt.plot(x, y, color='orange', linewidth=3, label='line')
        # 设置图表的标签和网格
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    # 定义鼠标回调函数
    def on_mouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            print(x, y)
        else:
            pass

    def print_point_coordinate(self, img):
        # 创建窗口并将鼠标回调函数与窗口绑定
        cv.namedWindow('Image', cv.WINDOW_NORMAL)
        cv.imshow("Image", img)
        cv.setMouseCallback('Image', self.on_mouse)
        cv.waitKey(0)
