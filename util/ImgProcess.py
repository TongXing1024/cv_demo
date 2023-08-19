# @Time    : 2023/8/11 16:39
# @Author  : TONGXING
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
matplotlib.use('TkAgg')

class ImageProcessC:
    # 直方图
    def draw_histogram(self,img):
        """
        绘制灰度图像的直方图
        :param img:
        :return:
        """
        # 灰度化
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # 计算直方图
        hist = cv.calcHist([gray], [0], None, [256], [0, 256])
        print(hist)
        print(type(hist))
        # 绘制直方图
        plt.figure()
        plt.title("Grayscale Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
        plt.plot(hist)
        plt.xlim([0, 256])
        plt.show()
    # 掩膜
    def mask(self,img):
        """
        掩膜操作
        :param img:
        :return:
        """
        # 创建掩膜
        mask = np.zeros(img.shape[:2], dtype="uint8")
        cv.rectangle(mask, (0, 90), (290, 450), 255, -1)
        cv.imshow("Mask", mask)
        # 应用掩膜
        masked = cv.bitwise_and(img, img, mask=mask)
        cv.imshow("Applying the Mask", masked)
        cv.waitKey(0)
    # 霍夫圆检测
    def hough_circle(self,img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转换为灰度图
        # 进行中值模糊，去噪点
        img_median = cv.medianBlur(gray, 7)
        # 霍夫圆检测
        circles = cv.HoughCircles(img_median,cv.HOUGH_GRADIENT,1,200,param1=100,param2=50,minRadius=0,maxRadius=100)
        if(circles is None):
            print("没有检测到圆")
        else:
            for i in circles[0, :]:
                # 绘制圆形
                print(i[0])
                cv.circle(img, (int(i[0]), int(i[1])), int(i[2]), (0, 255, 0), 2)
                # 绘制圆心
                cv.circle(img, (int(i[0]), int(i[1])), 2, (0, 0, 255), 3)
        cv.imshow("circles", img)
        cv.waitKey(0)
    # 灰度重心-每一行
    def get_center_row_point(self,img):
        # 如果是彩色图像，转换为灰度图像
        if len(img.shape) > 2:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            gray = img
        # 二值化
        ret, threshold = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
        # 获取图像高度和宽度
        height, width = threshold.shape
        # 初始化一个数组来存储每一行的灰度重心坐标
        row_centroids = []
        # 遍历图像的每一行
        for y in range(height):
            row = threshold[y, :]
            # 计算灰度重心
            total_intensity = np.sum(row)
            if total_intensity > 0:
                weighted_sum = np.sum(np.arange(width) * row)
                cX = weighted_sum / total_intensity
            else:
                cX = 0
            row_centroids.append(cX)
        # 在图像上绘制每一行的灰度重心
        for y, cX in enumerate(row_centroids):
            cv.circle(img, (int(cX), y), 1, (0, 0, 255), -1)
        # 显示图像
        cv.namedWindow('Image', cv.WINDOW_NORMAL)
        cv.imshow("Image", img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    # 灰度重心-全部
    def get_center_point(self,img):
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