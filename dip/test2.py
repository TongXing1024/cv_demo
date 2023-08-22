# @Time    : 2023/8/21 16:43
# @Author  : TONGXING

import matplotlib.pyplot as plt
import cv2 as cv
import matplotlib
import numpy as np
from util.ImgProcess import ImageProcessC

matplotlib.use('TkAgg')

# 定义上点列表，存储真实上点
up_points = []
# 定义上点列表，存储ROI（感兴趣区域）左上点
up_roi_points = []
# 定义上点列表，存储ROI（感兴趣区域）右下点
down_roi_points = []
# 定义下点列表，存储真实下点
down_points = []


def get_Line():
    """
    获取水面与陆地交界处的直线
    使用霍夫变换的方法
    :return:
    """
    # 读取图像
    img_origin = cv.imread('reservoir/shending.jpg', cv.IMREAD_COLOR)
    # 裁剪
    # img = img_origin[200:1000, 2000:]
    img = img_origin[600:1000, 2000:]
    # 转换为灰度图
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 高斯滤波
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    # 边缘检测
    edges = cv.Canny(blur, 100, 200)
    lines = cv.HoughLines(edges, 0.8, np.pi / 180, 145)
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
        cv.line(img, (x1, y1), (x2, y2), (0, 255, 0))
    # 显示图像
    plt.subplot(121)
    plt.imshow(img[:, :, ::-1])
    plt.title('img')
    plt.show()


def get_top_points(img):
    # 读取裁剪好的图像
    # img = cv.imread('reservoir/img555.jpg', cv.IMREAD_COLOR)
    # 转换为灰度图
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 高斯滤波
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    # 二值化
    ret, binary = cv.threshold(blur, 180, 255, cv.THRESH_BINARY)
    # 显示二值化的结果
    plt.subplot(121)
    plt.imshow(binary, cmap='gray')
    plt.title('binary')
    # 腐蚀 让图像中高亮部分收缩
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    binary = cv.erode(binary, kernel)
    # 膨胀  让图像中高亮部分扩张
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    binary = cv.dilate(binary, kernel)
    # 生成一个与原图像大小相同的全黑图像
    new_img = np.zeros(binary.shape, np.uint8)
    height, width = binary.shape
    height_value = int(height * 0.33)
    new_img[0:height_value, :] = binary[0:height_value, :]
    # 获取图像中的质心坐标  此处调用了ImageProcessC类
    img_pro = ImageProcessC()
    center_point = img_pro.get_center_point(new_img)
    # 在原图上画出找出的点，并显示
    cv.circle(img, (center_point[0][0], center_point[0][1]), 1, (0, 0, 255), 1)
    return center_point[0]
    # 显示图像
    # plt.subplot(122)
    # plt.imshow(img[:, :, ::-1])
    # plt.title('res')
    # plt.show()


def get_bottom_points():
    img = cv.imread('reservoir/img_roi.jpg', cv.IMREAD_COLOR)
    # 把裁剪好的图像转换为灰度图
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 高斯滤波
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    # 二值化
    ret, binary = cv.threshold(blur, 180, 255, cv.THRESH_BINARY)
    # 腐蚀 让图像中高亮部分收缩
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    erode = cv.erode(binary, kernel)
    # 膨胀  让图像中高亮部分扩张
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    dilate = cv.dilate(erode, kernel)
    # 生成一个与原图像大小相同的全黑图像
    new_img = np.zeros(dilate.shape, np.uint8)
    # 裁剪感兴趣区域拿到图片的2/3高度
    height, width = dilate.shape
    height_value = int(height * 0.66)
    new_img[height_value:, :] = dilate[height_value:, :]
    # 边缘检测
    edges = cv.Canny(new_img, 100, 200)
    # 显示图像
    plt.subplot(141)
    plt.imshow(img[:, :, ::-1])
    plt.title('img_origin')
    # 拿到非0点的坐标（过滤边缘前）
    points = np.argwhere(edges != 0)
    # 遍历每个点
    for point in points:
        # 判断此点是否先黑后白
        if dilate[point[0] - 2, point[1]] == 255:
            edges[point[0], point[1]] = 0
        else:
            edges[point[0], point[1]] = 255
    # 拿到非0点的坐标（过滤边缘后）
    points = np.argwhere(edges != 0)
    # 拿到第一列的最大值
    max_point = np.max(points[:, 0])
    print(max_point)
    # 判断是否有多个最大值，如果有，取中间的那个
    if len(np.argwhere(points[:, 0] == max_point)) > 1:
        # 拿到这些点的坐标
        max_points = points[np.argwhere(points[:, 0] == max_point)]
        # 取中间的那个
        max_point = max_points[len(max_points) // 2][0]
    else:
        max_point = max_point
    # 画出max_point
    cv.circle(img, (max_point[1], max_point[0]), 1, (0, 0, 255), 1)
    # 显示腐蚀膨胀的结果
    plt.subplot(142)
    plt.imshow(dilate, cmap='gray')
    plt.title('dilate')
    # 显示边缘
    plt.subplot(143)
    plt.imshow(edges, cmap='gray')
    plt.title('edges')
    # 把边缘点画到原图上
    plt.subplot(144)
    plt.imshow(img[:, :, ::-1])
    plt.title('res')
    plt.show()


def points_process():
    """
    此函数拿到yolo检测到的点以后，对其进行处理，使其符合画矩形的要求，提取出感兴趣区域
    """
    # 存储最终的坐标点
    res_points = []
    # 上点
    x_up = [185, 543, 821, 1050, 1198, 1405]
    y_up = [512, 552, 566, 576, 571, 609]
    points_up = np.column_stack((x_up, y_up))  # 合并 [185,512]
    # 下点
    x_down = [259, 603, 872, 1076, 1226, 1427]
    y_down = [749, 762, 749, 736, 742, 704]
    # 由于检测的坐标与真实的坐标有一定的偏差，所以需要对检测的坐标进行修正（下点的y坐标加30）
    y_down = [i + 30 for i in y_down]  # 下点的y坐标加30
    points_down = np.column_stack((x_down, y_down))
    # 处理这些点,使其符合画矩形的要求
    points_up[:, 0] -= 10
    points_down[:, 0] += 10
    # 把最终坐标添加到res_points中
    res_points.append(points_up)
    res_points.append(points_down)
    return res_points


def get_roi(img):
    """
    截取感兴趣区域
    img为水尺原图
    """
    res_points = points_process()
    # 获取第二根柱子的左上右下点坐标
    up_p = res_points[0][1]
    down_p = res_points[1][1]
    up_roi_points.append(up_p)
    down_roi_points.append(down_p)
    # 生成一个跟原图像大小相同的全黑图像
    # new_img = np.zeros(img.shape, np.uint8)
    # 裁剪
    new_img = img[up_p[1]:down_p[1], up_p[0]:down_p[0]]
    # 显示图像
    # cv.imshow('img', new_img)
    # cv.waitKey(0)
    return new_img


def draw_up_points():
    # 声明全局变量
    global up_points
    # 读取图像
    img = cv.imread('reservoir/img_origin.jpg', cv.IMREAD_COLOR)
    get_roi(img)
    # 读取裁剪好的图像
    img1 = cv.imread('reservoir/img_roi.jpg', cv.IMREAD_COLOR)
    # 获取上点
    up_points.append(get_top_points(img1))
    # 对应元素相加,获取点在全图的位置
    up_points = np.add(up_points, up_roi_points)
    print(up_points)
    # 画点
    cv.circle(img, (up_points[0][0], up_points[0][1]), 5, (0, 0, 255), 1)
    # 显示图像
    cv.imshow('img', img)
    cv.waitKey(0)


if __name__ == '__main__':
    draw_up_points()
