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

# 定义上点列表，存储真实上点
up_points = []
# 定义下点列表，存储真实下点
down_points = []


def get_center_point(img):
    # 创建数组保存质心坐标
    center_point = []
    # 如果是彩色图像，转换为灰度图像
    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        print("1111")
    else:
        img = img
    # 计算图像灰度重心
    M = cv.moments(img)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    center_point.append([cX, cY])
    return center_point


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
    """
    此函数用于获取水尺上的点
    img:裁剪好的图像
    返回值：水尺上的点
    """
    # 读取裁剪好的图像
    # img = cv.imread('reservoir/img555.jpg', cv.IMREAD_COLOR)
    # 如果是彩色图像，转换为灰度图像
    if len(img.shape) > 2:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img
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
    # 获取图像灰度重心
    center_point = []
    M = cv.moments(new_img)
    # 判断是否有质心
    if M["m00"] == 0:
        print('没有检测到质心')
        cX = 10
        cY = 20
    else:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    center_point.append([cX, cY])
    # 此时返回的值就是[x,y]坐标的形式，而不是索引的形式
    return center_point[0]


def points_process():
    """
    此函数拿到yolo检测到的点以后，对其进行处理，使其符合画矩形的要求，提取出感兴趣区域
    """
    # 定义上点列表，存储ROI（感兴趣区域）左上点
    up_roi_points = []
    # 定义上点列表，存储ROI（感兴趣区域）右下点
    down_roi_points = []
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
    # 把最终坐标添加到列表中
    up_roi_points.append(points_up)
    down_roi_points.append(points_down)
    # 转换成numpy数组
    up_roi_points = np.array(up_roi_points)
    down_roi_points = np.array(down_roi_points)
    # 注：此时返回的值就是[x,y]坐标的形式，而不是索引的形式
    return up_roi_points, down_roi_points


def get_roi(img, up_roi_points, down_roi_points):
    """
    此函数用于裁剪感兴趣区域
    img:原图像
    up_roi_points:上点
    down_roi_points:下点
    返回值：裁剪后的图像（矩阵数组）
    """
    # 定义一个矩阵数组，用于存储裁剪后的图像
    img_roi = []
    # 遍历每个点
    for i in range(len(up_roi_points[0])):
        # 裁剪
        img_roi.append(
            img[up_roi_points[0][i][1]:down_roi_points[0][i][1], up_roi_points[0][i][0]:down_roi_points[0][i][0]])
    # 显示裁剪后的图像
    # for i in range(len(img_roi)):
    #     cv.imshow('img_roi', img_roi[i])
    #     cv.waitKey(0)
    #     cv.destroyAllWindows()
    return img_roi


def draw_up_line():
    global up_points
    # 读取图像
    img = cv.imread('reservoir/img_origin.jpg', cv.IMREAD_COLOR)
    # 得到处理后ROI区域的点(左上，右下)
    up_roi_points, down_roi_points = points_process()
    # 得到裁剪后的图像
    img_roi = get_roi(img, up_roi_points, down_roi_points)
    # 遍历图像，画上点
    for i in range(len(img_roi)):
        up_points.append(get_top_points(img_roi[i]))
    # 转换成numpy数组
    up_points = np.array(up_points)

    print(up_points)
    # 对应元素相加,获取点在全图的位置
    up_points = np.add(up_points, up_roi_points[0])
    # 取第二个到第四个点
    up_points = up_points[1:4]
    # 把这些点拟合成直线
    up_lineParams = cv.fitLine(up_points, cv.DIST_L2, 0, 0.01, 0.01)
    a = up_lineParams[0]
    b = up_lineParams[1]  # (a,b)为直线的方向向量
    c = up_lineParams[2]
    d = up_lineParams[3]  # (c,d)为直线上的一点
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
    cv.line(img, (x1, y1), (x2, y2), (0, 255, 0))
    # 显示图像
    cv.imshow('img', img)
    cv.waitKey(0)


def get_down_points(img):
    if len(img.shape) > 2:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img
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
    # 遍历每个点的相邻点，此处的策略为：如果此点的正上方的点为白色，则此点为黑色，否则为白色
    for point in points:
        # 判断此点是否先黑后白
        if dilate[point[0] - 2, point[1]] == 255:
            edges[point[0], point[1]] = 0
        else:
            edges[point[0], point[1]] = 255
    # 拿到非0点的坐标（过滤边缘后）
    points = np.argwhere(edges != 0)
    # 拿到第一列的最大值（边缘线条的最低点）
    max_row_point = np.max(points[:, 0])
    # 判断是否有多个最大值，如果有，取中间的那个
    if len(np.argwhere(points[:, 0] == max_row_point)) > 1:
        # 拿到这些点的索引
        max_points = points[np.argwhere(points[:, 0] == max_row_point)]
        # 取中间的那个点的索引
        max_point = max_points[len(max_points) // 2]
        max_point = max_point[0]
    else:
        max_point = points[np.argwhere(points[:, 0] == max_row_point)]
        max_point = max_point[0][0]
    # 把max_point的第一列和第二列的值互换
    max_point = [max_point[1], max_point[0]]
    return max_point


def draw_down_line():
    global down_points
    # 读取图像
    img = cv.imread('reservoir/img_origin.jpg', cv.IMREAD_COLOR)
    # 得到处理后ROI区域的点(左上，右下)
    up_roi_points, down_roi_points = points_process()
    # 得到裁剪后的图像
    img_roi = get_roi(img, up_roi_points, down_roi_points)
    # 取前5个，因为最后一个检测不到，而且会引入许多噪声
    img_roi = img_roi[0:5]
    # 得到下点
    for i in range(len(img_roi)):
        down_points.append(get_down_points(img_roi[i]))
    # 转换成numpy数组
    down_points = np.array(down_points)
    # 对应上述前五个，这里也取前5个
    up_roi_points = up_roi_points[0][0:5]
    # 对应元素相加,获取点在全图的位置
    down_points = np.add(down_points, up_roi_points)
    # 取第二个到第四个点
    down_points = down_points[1:4]
    # 把这些点拟合成直线
    down_lineParams = cv.fitLine(down_points, cv.DIST_L2, 0, 0.01, 0.01)
    a = down_lineParams[0]
    b = down_lineParams[1]  # (a,b)为直线的方向向量
    c = down_lineParams[2]
    d = down_lineParams[3]  # (c,d)为直线上的一点
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
    cv.line(img, (x1, y1), (x2, y2), (0, 0, 255))
    # 显示图像
    cv.imshow('img', img)
    cv.waitKey(0)

if __name__ == '__main__':
    draw_up_line()
    # img = cv.imread('reservoir/img_origin.jpg', cv.IMREAD_COLOR)
    # up_roi_points, down_roi_points = points_process()
    # img_roi = get_roi(img, up_roi_points, down_roi_points)
    # # 取前5个
    # img_roi = img_roi[0:5]
    # # 遍历图像并显示
    # for i in range(len(img_roi)):
    #     down_points.append(get_down_points(img_roi[i]))
    # # 转换成numpy数组
    # down_points = np.array(down_points)
    # print(f"down_points:{down_points}")
    # # 取前5个
    # up_roi_points = up_roi_points[0][0:5]
    # print(f"up_roi_points:{up_roi_points}")
    # # 对应元素相加,获取点在全图的位置
    # down_points = np.add(down_points, up_roi_points)
    # print(f"add down_points:{down_points}")
    # # 画点
    # for i in range(len(down_points)):
    #     print(f"down_points[i]:{down_points[i]}")
    #     cv.circle(img, (down_points[i][0], down_points[i][1]), 10, (0, 0, 255), 1)
    # cv.namedWindow('img', cv.WINDOW_NORMAL)
    # cv.imshow('img', img)
    # cv.waitKey(0)
