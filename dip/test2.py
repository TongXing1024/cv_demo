# @Time    : 2023/8/21 16:43
# @Author  : TONGXING

import matplotlib.pyplot as plt
import cv2 as cv
import matplotlib
import numpy as np

np.set_printoptions(threshold=np.inf)
matplotlib.use('TkAgg')


def draw_line(img, lineParams, color):
    """
    此函数用于画出直线
    img:原图
    lineParams:直线参数
    color:直线颜色
    """
    a = lineParams[0]
    b = lineParams[1]  # (a,b)为直线的方向向量
    c = lineParams[2]
    d = lineParams[3]  # (c,d)为直线上的一点
    # 由方向向量和直线上的一点求直线的斜率
    k = b / a
    # 由方向向量和直线上的一点求直线的截距
    d = d - k * c
    print(f"故直线的方程为：y={k}x+{d}")
    # 随机取直线上的两点，画出直线
    x1 = 1
    y1 = int((k * x1 + d)[0])
    x2 = 2000
    y2 = int((k * x2 + d)[0])
    cv.line(img, (x1, y1), (x2, y2), color)


def get_Line():
    """
    获取水面与陆地交界处的直线
    使用霍夫变换的方法
    :return:
    """
    # 读取图像
    img_origin = cv.imread('reservoir/img_origin2.jpg', cv.IMREAD_COLOR)
    # 裁剪
    img = img_origin[550:1000, 1200:1800]
    # img = img_origin[600:1000, 2000:]
    # 转换为灰度图
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 高斯滤波
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    # 边缘检测
    edges = cv.Canny(blur, 100, 200)
    lines = cv.HoughLines(edges, 0.8, np.pi / 180, 165)
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
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255))
    # 显示图像
    plt.subplot(121)
    plt.imshow(img[:, :, ::-1])
    plt.title('img')
    plt.show()


def get_Line2():
    # 读取图像
    img_origin = cv.imread('reservoir/img_origin2.jpg', cv.IMREAD_COLOR)
    # 裁剪
    img = img_origin[550:1000, 1200:1800]
    # 转换为灰度图
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 高斯滤波
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    # 二值化
    ret, binary = cv.threshold(blur, 180, 255, cv.THRESH_BINARY)
    # 腐蚀 让图像中高亮部分收缩
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    binary = cv.erode(binary, kernel)
    # 膨胀  让图像中高亮部分扩张
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    binary = cv.dilate(binary, kernel)
    # 边缘检测
    edges = cv.Canny(binary, 100, 200)
    # 找到值不为0的点的坐标
    points = np.argwhere(edges != 0)
    # 取出第二列中最大的十个点
    points = points[np.lexsort(-points.T)][:10]
    # 交换列的位置
    points[:, [0, 1]] = points[:, [1, 0]]
    # 将点画在图像上
    for point in points:
        cv.circle(img, (point[0], point[1]), 5, (255, 0, 255), -1)
    # 拟合直线
    line_params = cv.fitLine(points, cv.DIST_L2, 0, 0.01, 0.01)
    # 返回直线参数
    return line_params

def get_line3():
    # 读取图像
    img = cv.imread('reservoir/img_origin2.jpg', cv.IMREAD_COLOR)
    # 转换为灰度图
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 高斯滤波
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    # 二值化
    ret, binary = cv.threshold(blur, 180, 255, cv.THRESH_BINARY)
    # 腐蚀 让图像中高亮部分收缩
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    binary = cv.erode(binary, kernel)
    # 膨胀  让图像中高亮部分扩张
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    binary = cv.dilate(binary, kernel)
    # 边缘检测
    edges = cv.Canny(binary, 100, 200)
    height, width = edges.shape
    height = int(height * 0.66)
    # 生成一个与原图像大小相同的全黑图像
    new_img = np.zeros(edges.shape, np.uint8)
    # 裁剪
    new_img[height:, :] = edges[height:, :]
    points = []
    # 遍历所有行
    for i in range(height, 0, -1):
        # 找到最后一个非0点的坐标
        point = np.argwhere(new_img[i, :] != 0)
        points.append(point)
    print(points)
    # 显示图像
    plt.subplot(121)
    plt.imshow(img[:, :, ::-1])
    plt.title('img')
    plt.subplot(122)
    plt.imshow(new_img, cmap='gray')
    plt.title('edges')
    plt.show()  

if __name__ == '__main__':
    get_line3()
