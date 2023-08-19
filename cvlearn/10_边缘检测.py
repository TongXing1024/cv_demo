import matplotlib
from my_cv_utils import cv_show
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
matplotlib.use('TkAgg')

def fun_sobel():
    img = cv.imread('../dip/img/horse.jpg', 0)
    #计算sobel卷积结果
    x = cv.Sobel(img,cv.CV_64F,1,0)
    print(f"x={x}")
    # x = cv.Sobel(img,cv.CV_16S,1,0,ksize=-1) ksize默认为3  如果设为-1 那么这就是scharr算子
    y = cv.Sobel(img,cv.CV_64F,0,1)
    print(f"y={y}")
    #将数据进行转换
    Scale_absX = cv.convertScaleAbs(x)
    Scale_absY = cv.convertScaleAbs(y)
    #显示结果
    cv.imshow('x',Scale_absX)
    cv.waitKey(0)
    #结果合成
    res = cv.addWeighted(Scale_absX,0.5,Scale_absY,0.5,0)
    plt.imshow(res,cmap=plt.cm.gray)
    plt.show()


def fun_lablacian():
    img = cv.imread('../dip/img/horse.jpg', 0)
    #lablacian转换
    res = cv.Laplacian(img,cv.CV_16S)
    Scale_abs = cv.convertScaleAbs(res)
    plt.figure(figsize=(10,8),dpi=100)
    plt.subplot(121),plt.imshow(img,cmap=plt.cm.gray),plt.title('origin')
    plt.figure(figsize=(10,8),dpi=100)
    plt.subplot(121),plt.imshow(Scale_abs,cmap=plt.cm.gray),plt.title('检测后的结果')
    plt.show()

def fun_canny():
    img = cv.imread('../dip/img/horse.jpg', 0)
    print(img.shape)
    #canny边缘检测
    low = 0
    max = 100
    canny = cv.Canny(img,low,max)
    plt.figure(figsize=(10, 8), dpi=100)
    plt.subplot(121), plt.imshow(img, cmap=plt.cm.gray), plt.title('origin')
    plt.figure(figsize=(10, 8), dpi=100)
    plt.subplot(121), plt.imshow(canny, cmap=plt.cm.gray), plt.title('canny')
    plt.show()




if __name__ == '__main__':
     fun_sobel()
    # fun_lablacian()

     # fun_canny()