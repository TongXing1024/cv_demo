"""
@Time ： 2023/7/17 15:53
@Author ： 佟星
"""
import matplotlib.pyplot as plt
import cv2 as cv
import matplotlib
import numpy as np
import sympy as sp
from scipy.io import loadmat
import tkinter as tk
from tkinter import messagebox
from tkinter import filedialog

matplotlib.use('TkAgg')
# 定义水准点D0
D0 = None
# 定义相邻水尺间距Δ
deerta = None  # 水准点D0和相邻水尺间距Δ 由输入框人工提供

# 定义上点列表，存储真实上点
up_points = []
# 定义下点列表，存储真实下点
down_points = []
mat = loadmat('mat/points.mat')['points']
x_up = mat[0]
y_up = mat[1]
x_down = mat[2]
y_down = mat[3]


# # 上点(检测)
# x_up = [185, 543, 821, 1050, 1198, 1405]
# y_up = [512, 552, 566, 576, 571, 609]
# # 下点(检测)
# x_down = [259, 603, 872, 1076, 1226, 1427]
# y_down = [749, 762, 749, 736, 742, 704]

# 此函数用于获取灰度重心
def get_gray_center_point(img):
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


def get_up_points(img):
    """
    此函数用于获取水尺上的点
    img:裁剪好的图像
    返回值：水尺上点
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
        cX = 10
        cY = 20
    else:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    center_point.append([cX, cY])
    # 此时返回的值就是[x,y]坐标的形式，而不是索引的形式
    return center_point[0]


def points_process():
    global x_up, y_up, x_down, y_down
    """
    此函数拿到yolo检测到的点以后，对其进行处理，使其符合画矩形的要求，提取出感兴趣区域
    """
    # 定义上点列表，存储ROI（感兴趣区域）左上点
    up_roi_points = []
    # 定义上点列表，存储ROI（感兴趣区域）右下点
    down_roi_points = []

    points_up = np.column_stack((x_up, y_up))  # 合并 [185,512]

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
    return img_roi


def get_up_lineParams(img):
    global up_points
    # 得到处理后ROI区域的点(左上，右下)
    up_roi_points, down_roi_points = points_process()
    # 得到裁剪后的图像
    img_roi = get_roi(img, up_roi_points, down_roi_points)
    # 遍历图像，添加上点
    for i in range(len(img_roi)):
        up_points.append(get_up_points(img_roi[i]))
    # 转换成numpy数组
    up_points = np.array(up_points)
    # 对应元素相加,获取点在全图的位置
    up_points = np.add(up_points, up_roi_points[0])
    # 取第二个到最后一个点
    up_points = up_points[1:]
    # 把这些点拟合成直线
    up_lineParams = cv.fitLine(up_points, cv.DIST_L2, 0, 0.01, 0.01)
    k = up_lineParams[1] / up_lineParams[0]
    b = up_lineParams[3] - k * up_lineParams[2]
    up_lineParams = [k[0], b[0]]
    # 转换成numpy数组
    up_lineParams = np.array(up_lineParams)
    # 返回直线参数
    return up_lineParams


def get_down_lineParams(img):
    global down_points
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
    down_line_Params = cv.fitLine(down_points, cv.DIST_L2, 0, 0.01, 0.01)
    k = down_line_Params[1] / down_line_Params[0]
    b = down_line_Params[3] - k * down_line_Params[2]
    down_line_Params = [k[0], b[0]]
    # 转换成numpy数组
    down_line_Params = np.array(down_line_Params)
    # 返回直线参数
    return down_line_Params


def get_boundary_lineParams(img):
    """
    此函数用于获取水陆交界线参数
    img:原图像
    """
    global y_up, y_down
    # 取出y_up的最小值
    max_y_up = np.min(y_up)
    # 取出y_down的最大值
    min_y_down = np.max(y_down)
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
    # 生成两个与原图像大小相同的全黑图像
    new_img1 = np.zeros(edges.shape, np.uint8)
    new_img2 = np.zeros(edges.shape, np.uint8)
    # 裁剪
    new_img1[max_y_up - 50:max_y_up, :] = edges[max_y_up - 50:max_y_up, :]
    new_img2[min_y_down:min_y_down + 100, :] = edges[min_y_down:min_y_down + 100, :]
    # 创建空numpy数组
    points = []
    # 遍历裁剪好的图像的每一行
    for i in range(max_y_up - 100, max_y_up):
        # 拿到每一行的非0点的坐标，而且y坐标最大的那个点
        point = np.argwhere(new_img1[i, :] != 0)
        if len(point) > 0:
            point = point[np.lexsort(-point.T)][0]
            points.append([i, point[0]])
    for i in range(min_y_down, min_y_down + 100):
        # 拿到每一行的非0点的坐标，而且y坐标最大的那个点
        point = np.argwhere(new_img2[i, :] != 0)
        if len(point) > 0:
            point = point[np.lexsort(-point.T)][0]
            points.append([i, point[0]])
    # 转换成numpy数组
    points = np.array(points)
    # 将点画在图像上
    # for point in points:
    #     cv.circle(img, (point[1], point[0]), 5, (255, 0, 255), -1)

    # 交换第一列和第二列的位置
    points[:, [0, 1]] = points[:, [1, 0]]
    # 拟合曲线
    params = np.polyfit(points[:, 0], points[:, 1], 2)
    # print(f"曲线方程为：y={params[0]}x^2+{params[1]}x+{params[2]}")
    # 画线
    x = np.arange(0, 2000)
    y = params[0] * x ** 2 + params[1] * x + params[2]
    # plt.plot(x, y, 'r')
    # 返回曲线参数
    return params


def draw_straight_line(lineParams, color):
    """
    此函数用于画出直线
    img:原图
    lineParams:直线参数
    color:直线颜色
    """
    # 由方向向量和直线上的一点求直线的斜率
    k = lineParams[0]
    # 由方向向量和直线上的一点求直线的截距
    b = lineParams[1]
    # 随机取直线上的两点，画出直线
    x1 = 1
    y1 = int((k * x1 + b))
    x2 = 2300
    y2 = int((k * x2 + b))
    # 画线
    plt.plot([x1, x2], [y1, y2], color)


def get_intersection_point(straight_lineParams, boundary_lineParams):
    """
    此函数用于求出直线与曲线的交点
    params1:直线参数1
    params2:曲线参数2
    """
    # 直线参数 y = kx + b
    k = straight_lineParams[0]
    b1 = straight_lineParams[1]
    # print(f"直线方程为：y={k}x+{b1}")
    # 曲线参数  y = ax^2 + bx + c
    a, b2, c = boundary_lineParams[0], boundary_lineParams[1], boundary_lineParams[2]
    # print(f"曲线方程为：y={a}x^2+{b2}x+{c}")
    # 定义变量
    x = sp.symbols('x')
    # 合并同类项
    eq_linear = a * x ** 2 + (b2 - k) * x + (c - b1)
    solution_linear = sp.solve(eq_linear, x)
    # 找到大于0的解
    for i in range(len(solution_linear)):
        if solution_linear[i] > 0:
            solution_linear = solution_linear[i]
            break
    x = solution_linear
    y = k * x + b1
    intersection_point = np.array([x, y])
    return intersection_point


def has_chinese_characters(text):
    """
    判断是否含有中文
    :param text:
    :return:
    """
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False


def select_file_dialog():
    """
    此函数用于选择图片路径
    :return: 图片路径、字符串形式
    """
    tk.messagebox.showinfo("提示", "请选择水尺图片")
    file_path = filedialog.askopenfilename()
    if file_path:
        if has_chinese_characters(file_path):
            # 文件路径包含中文字符，弹出警告对话框
            tk.messagebox.showwarning("警告", "所选文件路径包含中文字符，请把文件放入英文文件夹下边")
        return file_path
    else:
        return None


def open_input_dialog():
    """
    此函数用于打开输入窗口,输入水尺高程D0和相邻水尺间距Δ
    """

    def submit():
        global D0, deerta
        try:
            D0 = float(entry1.get())
            deerta = float(entry2.get())
            # 关闭窗口
            root.destroy()
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数字")

    # 创建主窗口
    root = tk.Tk()
    root.title("输入窗口")

    # 创建标签和输入框1
    label1 = tk.Label(root, text="请输入水尺高程D0，厘米cm为单位：")
    label1.pack()
    entry1 = tk.Entry(root)
    entry1.pack()

    # 创建标签和输入框2
    label2 = tk.Label(root, text="请输入相邻水尺的间距Δ，厘米cm为单位：")
    label2.pack()
    entry2 = tk.Entry(root)
    entry2.pack()

    # 创建提交按钮
    submit_button = tk.Button(root, text="提交", command=submit)
    submit_button.pack()

    # 设置窗口大小和位置
    root.geometry("300x200+500+200")
    # 进入主循环
    root.mainloop()


def get_points_distance(point1, point2):
    """
    此函数计算两点之间的距离
    """
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def get_water_level(D0, deerta):
    """
    d = D0 - Δ*(i - 1 - l/T)
    """
    global x_down, y_down
    # 水尺个数
    i = len(x_down)
    # 合并上下点
    points = np.column_stack((x_down, y_down))
    # 取出倒数第二个
    points1 = points[-2]
    # 取出倒数第三个
    points2 = points[-3]
    # 计算两点之间的距离T
    T = get_points_distance(points1, points2)
    print(T)



def draw_all_things(img, D0, deerta):
    global x_down, y_down
    # 水尺个数
    i = len(x_down)
    # 合并上下点
    points = np.column_stack((x_down, y_down))
    # 取出倒数第二个
    points1 = points[-2]
    # 取出倒数第三个
    points2 = points[-3]
    # 拿到下点的直线参数
    down_lineParams = get_down_lineParams(img)
    draw_straight_line(down_lineParams, 'blue')
    # 拿到上点的直线参数
    up_lineParams = get_up_lineParams(img)
    draw_straight_line(up_lineParams, 'green')
    # 拿到水陆交界线的曲线参数
    boundary_lineParams = get_boundary_lineParams(img)
    # 画出水陆交界线
    x = np.arange(0, 2000)
    y = boundary_lineParams[0] * x ** 2 + boundary_lineParams[1] * x + boundary_lineParams[2]
    plt.plot(x, y, 'yellow')
    # 拿到下线与水陆交界线的交点
    down_intersection_point = get_intersection_point(down_lineParams, boundary_lineParams)
    print("下点交点：", down_intersection_point)
    # 四舍五入取整
    down_intersection_point = np.array(down_intersection_point, dtype=np.float32)
    down_intersection_point = np.round(down_intersection_point)
    # 计算相邻两直线之间的距离T
    T = get_points_distance(points1, points2)
    print(f"相邻两直线之间的距离T为：{T}")
    # 计算交点到水尺距离l
    l = get_points_distance(down_intersection_point, points1)
    print("距离：", l)
    # 计算水位
    d = D0 - deerta * (i - 1 - l / T)
    print("D0:", D0)
    print("deerta:", deerta)
    print(f"水位为：{d}")
    # 在图片上写字
    cv.putText(img, f"water_level:{d}cm", (100,100), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    # 画交点
    cv.circle(img, (int(down_intersection_point[0]), int(down_intersection_point[1])), 10, (255, 0, 255), -1)
    plt.imshow(img[:, :, ::-1])
    plt.show()


if __name__ == '__main__':
    open_input_dialog()
    # # 选择图片
    # selected_file = select_file_dialog()
    # # 读取图像
    # img = cv.imread(selected_file, cv.IMREAD_COLOR)
    # 画出所有的东西
    img = cv.imread('reservoir/img_origin2.jpg', cv.IMREAD_COLOR)
    draw_all_things(img, D0, deerta)
