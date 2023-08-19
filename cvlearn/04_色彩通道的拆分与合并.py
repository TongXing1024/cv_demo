# Author : 佟星TONGXING
# Date : 2022/11/24 21:26
# Version: 1.0
import cv2 as cv
from util.my_cv_utils import cv_show

def color_split(img):
    cv_show(img)
    #通道拆分
    b,g,r = cv.split(img)

    #通道合并
    img1 = cv.merge((b,g,r))
    cv_show(b)

if __name__ == '__main__':
    img = cv.imread("images/food.jpg", cv.IMREAD_COLOR)
    color_split(img)