# Author : 佟星TONGXING
# Date : 2022/11/24 20:45
# Version: 1.0
import cv2 as cv
import numpy as np

def update_pixel_val():
    img = np.zeros((512,512,3),np.uint8)
    cv.imshow('img',img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    #获取图像属性
    print(img.shape)
    #获取图像的像素数
    print(f"图像的像素个数为：{img.size}")
    #获取某个像素点的值
    px = img[100,100]
    print(px)
    #仅获取蓝色通道的强度值
    #blue = img[100,100,0]
    blue = img[100][100][0]
    #修改某个点的像素值
    img[100,100] = [0,255,255]
    cv.imshow('img', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    update_pixel_val()