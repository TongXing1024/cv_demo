# @Time    : 2023/8/15 17:21
# @Author  : TONGXING
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from util.Draw import DrawC
matplotlib.use('TkAgg')

def draw_line():
    # 读取图像
    img_x_y = cv.imread('reservoir/img_xy1.jpg', cv.IMREAD_COLOR)
    # 检测坐标点
    # draw = Draw()
    # draw.print_point_coordinate(img_x_y)
    # 水尺上坐标点
    x_up = [192, 1421]
    y_up = [551, 634]
    points_up = np.column_stack((x_up, y_up))
    # 水尺下坐标点
    x_down = [269, 1414]
    y_down = [791, 741]
    points_down = np.column_stack((x_down, y_down))
    # 陆地水面坐标点
    x_row = [1012, 1461]
    y_row = [596, 819]
    points_row = np.column_stack((x_row, y_row))
    # 画出直线
    cv.line(img_x_y, (points_up[0][0], points_up[0][1]), (points_up[1][0], points_up[1][1]), (0, 255, 0), 2)
    cv.line(img_x_y, (points_down[0][0], points_down[0][1]), (points_down[1][0], points_down[1][1]), (0, 255, 0), 2)
    cv.line(img_x_y, (points_row[0][0], points_row[0][1]), (points_row[1][0], points_row[1][1]), (255, 0, 255), 2)
    # 输出结果
    # 输出文字
    # font = cv.FONT_HERSHEY_SIMPLEX
    # cv.putText(img_x_y, '识别结果\n：水位高度为:D - 85cm\n 注：D为水准点高度', (10, 500), font, 1, (255, 255, 255), 2, cv.LINE_AA)
    cv.imshow('img', img_x_y)
    # 保存图像
    cv.imwrite('reservoir/img_x_y_line.jpg', img_x_y)

    cv.waitKey(0)
def get_real_coordinate(img):
    draw = DrawC()
    draw.print_point_coordinate(img)


def get_global_coordinate_1(img):
    """
    根据第一次的结果
    """
    # 转换为灰度图
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 上点
    x_up = [185, 543, 821, 1198, 1405]
    y_up = [489, 529, 543, 548, 586]
    points_up = np.column_stack((x_up, y_up))
    # 画出这些点
    for i in range(len(points_up)):
        cv.circle(img, (points_up[i][0], points_up[i][1]), 5, (0, 0, 255), -1)
    # 下点
    x_down = [259, 603, 872, 1226, 1427]
    y_down = [726, 739, 726, 719, 681]
    points_down = np.column_stack((x_down, y_down))
    # 画出这些点
    for i in range(len(points_down)):
        cv.circle(img, (points_down[i][0], points_down[i][1]), 5, (0, 0, 255), -1)
    # 由于检测的坐标与真实的坐标有一定的偏差，所以需要对检测的坐标进行修正
    y_up = [i + 20 for i in y_up]
    y_down = [i + 50 for i in y_down]
    # 画出这些点
    for i in range(len(points_up)):
        cv.circle(img, (x_up[i], y_up[i]), 5, (0, 255, 0), -1)
    for i in range(len(points_down)):
        cv.circle(img, (x_down[i], y_down[i]), 5, (0, 255, 0), -1)
    # 画矩形
    for i in range(len(points_up)):
        cv.rectangle(img, (x_up[i]-10, y_up[i]), (x_down[i]+10, y_down[i]), (0, 255, 255), 2)
    # 显示图像
    cv.namedWindow('img', cv.WINDOW_NORMAL)
    cv.imshow('img', img)
    cv.waitKey(0)

def points_process(img):
    """
    根据第二次结果  new_img_xy
    """
    # 转换为灰度图
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 上点
    x_up = [185, 543, 821,1050,1198,1405]
    y_up = [512, 552, 566,576,571,609]
    points_up = np.column_stack((x_up, y_up))
    # 画出这些点
    for i in range(len(points_up)):
        cv.circle(img, (points_up[i][0], points_up[i][1]), 5, (0, 0, 255), -1)
    # 下点
    x_down = [259, 603, 872, 1076,1226,1427]
    y_down = [749, 762, 749, 736,742,704]
    points_down = np.column_stack((x_down, y_down))
    # 画出这些点
    for i in range(len(points_down)):
        cv.circle(img, (points_down[i][0], points_down[i][1]), 5, (0, 0, 255), -1)
    # 由于检测的坐标与真实的坐标有一定的偏差，所以需要对检测的坐标进行修正
    # y_up = [i + 20 for i in y_up]
    y_down = [i + 30 for i in y_down]
    # 画出这些点
    for i in range(len(points_up)):
        cv.circle(img, (x_up[i], y_up[i]), 5, (0, 255, 0), -1)
    for i in range(len(points_down)):
        cv.circle(img, (x_down[i], y_down[i]), 5, (0, 255, 0), -1)
    # 画矩形
    for i in range(len(points_up)):
        cv.rectangle(img, (x_up[i]-10, y_up[i]), (x_down[i]+10, y_down[i]), (0, 255, 255), 2)
    # 显示图像
    cv.namedWindow('img', cv.WINDOW_NORMAL)
    cv.imshow('img', img)
    cv.waitKey(0)



if __name__ == '__main__':
    img = cv.imread('reservoir/img_origin.jpg', cv.IMREAD_COLOR)
    get_global_coordinate_1(img)
    # get_real_coordinate(img)


