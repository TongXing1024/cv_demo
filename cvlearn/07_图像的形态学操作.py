import matplotlib
from my_cv_utils import cv_show
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

matplotlib.use("TkAgg")

#腐蚀和膨胀
def image_erode():
    img = cv.imread('line_b_w.png',cv.IMREAD_COLOR)
    #创建核结构
    kernel = np.ones((5,5),np.uint8)
    # 腐蚀
    res1 = cv.erode(img,kernel)
    #膨胀
    res2 = cv.dilate(img,kernel)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8), dpi=100)
    axes[0].imshow(img[:, :, ::-1])
    axes[0].set_title("origin")
    axes[1].imshow(res1[:, :, ::-1])
    axes[1].set_title("erode")
    axes[2].imshow(res2[:, :, ::-1])
    axes[2].set_title("enlarge")
    plt.show()

#开闭运算
def image_morphology():
    img = cv.imread('line_b_w.png', cv.IMREAD_COLOR)
    # 创建核结构
    kernel = np.ones((10, 10), np.uint8)
    #开运算  先腐蚀后膨胀
    imgOpen = cv.morphologyEx(img,cv.MORPH_OPEN,kernel)
    #闭运算  先膨胀后腐蚀
    imgClose = cv.morphologyEx(img,cv.MORPH_CLOSE,kernel)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 8), dpi=100)
    axes[0].imshow(img[:, :, ::-1])
    axes[0].set_title("origin")
    axes[1].imshow(imgOpen[:, :, ::-1])
    axes[1].set_title("imgOpen")
    axes[2].imshow(imgClose[:, :, ::-1])
    axes[2].set_title("imgClose")
    plt.show()

# 礼帽和黑帽
def image_hat():
    img = cv.imread('line_b_w.png', cv.IMREAD_COLOR)
    # 创建核结构
    kernel = np.ones((10, 10), np.uint8)
    # 礼帽 原图-开运算图
    imgOpen = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
    #黑帽 闭运算图-原图
    imgClose = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 8), dpi=100)
    axes[0].imshow(img[:, :, ::-1])
    axes[0].set_title("origin")
    axes[1].imshow(imgOpen[:, :, ::-1])
    axes[1].set_title("imgTopHat")
    axes[2].imshow(imgClose[:, :, ::-1])
    axes[2].set_title("imgBlackHat")
    plt.show()
if __name__ == '__main__':
    #膨胀和腐蚀
    image_erode()

    #开闭运算
    # image_morphology()

    #帽运算
    # image_hat()