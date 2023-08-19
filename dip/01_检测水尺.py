"""
@Time ： 2023/7/17 15:53
@Author ： 佟星
"""
import matplotlib.pyplot as plt
import cv2 as cv
import matplotlib
import numpy as np
from util.ImgProcess import ImageProcessC

matplotlib.use('TkAgg')


def draw_line():
    # 读取图像
    img_x_y = cv.imread('reservoir/img_x_y.jpeg', cv.IMREAD_COLOR)
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


def img_test():
    """
    此方法用于测试图片裁剪的尺寸
    :return:
    """
    # 读取图像
    img_origin = cv.imread('reservoir/shending.jpg', cv.IMREAD_COLOR)
    # 裁剪
    # img = img_origin[400:1000, 2000:]
    img = img_origin[600:1000, 2000:]
    # 转换为灰度图
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 高斯滤波
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    edges = cv.Canny(blur, 100, 200)
    # 显示图像
    plt.subplot(221)
    plt.imshow(img_origin[:, :, ::-1])
    plt.title('img_origin')
    plt.subplot(222)
    plt.imshow(img[:, :, ::-1])
    plt.title('img')

    plt.subplot(223)
    plt.imshow(edges, cmap='gray')
    plt.show()


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


def draw_on_img():
    #1
    # 读取图像
    # 测
    img_origin = cv.imread('reservoir/img_origin.jpg', cv.IMREAD_COLOR)
    # # 画线
    # cv.line(img_origin, (185, 489), (1198, 548), (0, 255, 0))
    # cv.imshow('img', img_origin)
    # cv.waitKey(0)
    x_up = [185, 543, 1198, 1405]
    y_up = [489, 529, 548, 586]
    points_up = np.column_stack((x_up, y_up))
    # 画出这些点
    for i in range(len(points_up)):
        cv.circle(img_origin, (points_up[i][0], points_up[i][1]), 5, (0, 0, 255), -1)
    cv.imshow('img', img_origin)
    cv.waitKey(0)

    x_down = [259, 603, 872, 1226]
    y_down = [726, 739, 726, 719]
    points_down = np.column_stack((x_down, y_down))


def get_top_points():
    # 读取裁剪好的图像
    img = cv.imread('reservoir/img555.jpg', cv.IMREAD_COLOR)
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
    new_img[0:20, :] = binary[0:20, :]

    # 获取图像中的质心坐标
    img_pro = ImageProcessC()
    center_point = img_pro.get_center_point(new_img)
    # 在原图上画出找出的点，并显示
    cv.circle(img, (center_point[0][0], center_point[0][1]), 1, (0, 0, 255), 1)
    plt.subplot(122)
    plt.imshow(img[:, :, ::-1])
    plt.title('res')
    plt.show()


def get_bottom_points():
    # 读取裁剪好的图像
    img = cv.imread('reservoir/img555.jpg', cv.IMREAD_COLOR)
    # 转换为灰度图
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
    # 裁剪感兴趣区域  拿到图片的2/3高度
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
        if dilate[point[0]-2, point[1]] == 255:
            edges[point[0], point[1]] = 0
        else:
            edges[point[0], point[1]] = 255
    # 拿到非0点的坐标（过滤边缘后）
    points = np.argwhere(edges != 0)

    # 拿到第一列的最大值
    max_point = np.max(points[:, 0])
    # 判断是否有多个最大值，如果有，取中间的那个
    if len(np.argwhere(points[:, 0] == max_point)) > 1:
        # 拿到这写点的坐标
        max_points = points[np.argwhere(points[:, 0] == max_point)]
        # 取中间的那个
        max_point = max_points[len(max_points) // 2][0]
        # 输出最大值
        print(max_point)
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


# 调用函数
get_bottom_points()

# 获取水尺上方的点
# get_top_points()

# 画出渐近线
# draw_line()

# 获取陆水交界线
# get_Line()

# 测试
# img_test()


# draw_on_img()
