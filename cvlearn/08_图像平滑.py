import matplotlib
from my_cv_utils import cv_show
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
matplotlib.use('TkAgg')


#均值滤波
def image_blur():
    img = cv.imread('images/dogsp.jpeg', cv.IMREAD_COLOR)
    # 均值滤波
    blur = cv.blur(img,(5,5))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8), dpi=100)
    axes[0].imshow(img[:, :, ::-1])
    axes[0].set_title("origin")
    axes[1].imshow(blur[:, :, ::-1])
    axes[1].set_title("blur")
    plt.show()

#高斯滤波
def image_gauss():
    img = cv.imread('images/dogGauss.jpeg', cv.IMREAD_COLOR)
    # 高斯滤波
    blur = cv.GaussianBlur(img,(3,3),1)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8), dpi=100)
    axes[0].imshow(img[:, :, ::-1])
    axes[0].set_title("origin")
    axes[1].imshow(blur[:, :, ::-1])
    axes[1].set_title("blur")
    plt.show()

#中值滤波
def image_median_blur():
    img = cv.imread('images/dogsp.jpeg', cv.IMREAD_COLOR)
    blur = cv.medianBlur(img,5)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8), dpi=100)
    axes[0].imshow(img[:, :, ::-1])
    axes[0].set_title("origin")
    axes[1].imshow(blur[:, :, ::-1])
    axes[1].set_title("blur")
    plt.show()


if __name__ == '__main__':
    # 均值滤波
    # image_blur()

    # 高斯滤波
    #  image_gauss()

    # 中值滤波
    image_median_blur()