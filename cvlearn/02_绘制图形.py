# Author : 佟星TONGXING
# Date : 2022/11/24 20:19
# Version: 1.0
import cv2 as cv
import numpy as np


def draw():
    #1 创建一个空白图像
    img = np.zeros((512,512,3),np.uint8)

    #2 在图像上绘制图形
    # 线条
    cv.line(img,(0,0),(512,512),(255,0,0),5)

    #矩形
    cv.rectangle(img,(100,100),(400,400),(0,255,0),3)

    #圆形
    cv.circle(img,(256,256),60,(0,0,255),-1)

    #多边形
    pts = np.array([[10, 5], [20, 30], [70, 20], [50, 10]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv.polylines(img, [pts], True, (0, 255, 255))


    #文字
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(img,"we are young",(10,500),font,1,(255,255,255),2,cv.LINE_AA)

    #图像展示
    cv.imshow('img',img)
    cv.waitKey(0)
    cv.destroyAllWindows()
if __name__ == '__main__':
    draw()