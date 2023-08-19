"""
@Time ： 2023/6/26 17:00
@Author ： 佟星
"""
import matplotlib
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np



# 读取视频
cap = cv.VideoCapture('carPark.mp4')
print(cap.isOpened())
# 判断视频是否打开
while cap.isOpened():
    # 读取视频的帧
    ret, frame = cap.read()
    if ret:
        # 显示视频
        cv.imshow('frame', frame)
        # 按下q键退出
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# 释放资源
cap.release()
cv.destroyAllWindows()
