# @Time    : 2023/9/4 20:48
# @Author  : TONGXING
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
def get_boundary_lineParams(img):
    # 取出y_down的最大值
    max_y_down = 1053
    # 转换为灰度图
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 高斯滤波
    blur = cv.GaussianBlur(gray, (3, 3), 0)
    # 二值化
    ret, binary = cv.threshold(blur, 100, 155, cv.THRESH_BINARY)
    # 腐蚀 让图像中高亮部分收缩
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    binary = cv.erode(binary, kernel)
    # 膨胀  让图像中高亮部分扩张
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    binary = cv.dilate(binary, kernel)
    # 边缘检测
    edges = cv.Canny(binary, 80, 100)
    # 生成一个与原图像大小相同的全黑图像
    new_img = np.zeros(edges.shape, np.uint8)
    # 裁剪
    new_img[max_y_down:max_y_down + 100, :] = edges[max_y_down:max_y_down + 100, :]
    # cv.namedWindow("image", cv.WINDOW_NORMAL)
    # cv.imshow("image", new_img)
    # cv.waitKey(0)
    # 创建空numpy数组
    points = []
    # 遍历裁剪好的图像的每一行
    for i in range(max_y_down, max_y_down + 100):
        # 拿到每一行的非0点的坐标，而且y坐标最大的那个点
        point = np.argwhere(new_img[i, :] != 0)
        if len(point) > 0:
            point = point[np.lexsort(-point.T)][0]
            points.append([i, point[0]])
    # 转换成numpy数组
    points = np.array(points)
    # 将点画在图像上
    # for point in points:
    #     cv.circle(img, (point[1], point[0]), 5, (255, 0, 255), -1)
    # # 显示
    # cv.namedWindow("image", cv.WINDOW_NORMAL)
    # cv.imshow("image", img)
    # cv.waitKey(0).
    # 交换第一列和第二列的位置
    points[:, [0, 1]] = points[:, [1, 0]]
    # print(points)
    # 拟合直线
    params = cv.fitLine(points, cv.DIST_L2, 0, 0.01, 0.01)
    k = params[1] / params[0]
    b = params[3] - k * params[2]
    params = [k[0], b[0]]
    # 转换成numpy数组
    params = np.array(params)
    return params




def draw(img):
    params = get_boundary_lineParams(img)
    a1 = (129, 704)
    b1 = (1489, 647)
    a2 = (144, 1053)
    b2 = (1237, 924)
    # 由两点坐标计算直线方程
    k1 = (b1[1] - a1[1]) / (b1[0] - a1[0])
    k2 = (b2[1] - a2[1]) / (b2[0] - a2[0])
    b1 = a1[1] - k1 * a1[0]
    b2 = a2[1] - k2 * a2[0]

    # 读取视频 画线
    cap = cv.VideoCapture("./reservoir/shendingmv.mp4")
    while True:
        ret, img = cap.read()
        if ret == False:
            break
        cv.line(img, (0, int(b1)), (img.shape[1], int(k1 * img.shape[1] + b1)), (0, 0, 255), 3)
        cv.line(img, (0, int(b2)), (img.shape[1], int(k2 * img.shape[1] + b2)), (0, 255,0 ), 3)
        cv.line(img, (0, int(params[1])), (img.shape[1], int(params[0] * img.shape[1] + params[1])), (255, 0, 255), 3)
        # 保存视频
        
        cv.namedWindow("image", cv.WINDOW_NORMAL)
        cv.imshow("image", img)
        c = cv.waitKey(50)
        # 按ESC退出
        if c == 27:
            break

def on(img):
    # 定义鼠标回调函数
    def on_mouse(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            print(x, y)
        else:
            pass

    # 创建窗口并将鼠标回调函数与窗口绑定
    cv.namedWindow('Image', cv.WINDOW_NORMAL)
    cv.imshow("Image", img)
    cv.setMouseCallback('Image', on_mouse)
    cv.waitKey(0)

if __name__ == '__main__':
    img = cv.imread("./reservoir/shending_origin.jpg")
    draw(img)
