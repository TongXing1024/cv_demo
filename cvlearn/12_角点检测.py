import matplotlib

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
matplotlib.use('TkAgg')

def fun_harris():
    img = cv.imread('images/chessboard.jpg', cv.IMREAD_COLOR)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #角点检测，输入图像必须是 float32
    gray = np.float32(gray)
    #最后一个参数在0.04到0.05之间
    dst = cv.cornerHarris(gray,2,3,0.04)
    #设置阈值，将焦点绘制出来，阈值根据图像进行选择
    img[dst>0.001*dst.max()] = [0,255,0]
    # 显示图形
    plt.imshow(img[:, :, ::-1])
    plt.title('result')
    plt.show()

def fun_tomasi():
    img = cv.imread('images/tv.jpg', cv.IMREAD_COLOR)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #角点检测
    corners = cv.goodFeaturesToTrack(gray,1000,0.01,10)
    print(corners)
    #绘制角点
    for i in corners:
        x,y = i.ravel()
        cv.circle(img,(int(x),int(y)),2,(0,0,255),1)
    # 显示图形
    plt.imshow(img[:, :, ::-1])
    plt.title('result')
    plt.show()

if __name__ == '__main__':
    #
    # fun_harris()
    #
    fun_tomasi()