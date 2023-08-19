"""
@Time ： 2023/3/21 20:38
@Author ： 佟星
"""
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from cvlearn.my_cv_utils import plt_show

matplotlib.use('TkAgg')
"""
重心横坐标 x = Σ(iw)/Σw

重心纵坐标 y = Σ(jw)/Σw
"""
def center_of_mass(img):
    rows, cols = img.shape
    row_sum = 0
    col_sum = 0
    for i in range(rows):
        for j in range(cols):
            row_sum += i * img[i][j]
            col_sum += j * img[i][j]
    row_com = row_sum / img.sum()
    col_com = col_sum / img.sum()
    return row_com, col_com

# # read binary image
# img = cv.imread('../light/Image_20230301205338890.bmp', 0)
#
# # calculate center of mass of image
# row_com, col_com = center_of_mass(img)
#
# print("Center of mass of binary image:", row_com, col_com)
#
# # read binary image
# img = cv.imread('binary_image.png', 0)
#
# # calculate sum of all pixel values
# sum_pixels = img.sum()
#
# print("Sum of all pixel values in binary image:", sum_pixels)

'146.53953752101165 -137.98317841529158'
# def extract(imgPath):
#     img = cv.imread(imgPath)
#     img1 = img[400:,1500:1700]
#     gray = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
#     # cv.imshow('img',img1)
#     # cv.waitKey(0)
#     rows,cols = gray.shape
#     print(f"gary:{gray}")
#     row_sum = 0
#     col_sum = 0
#     w = 0
#     for i in range(rows):
#         for j in range(cols):
#             row_sum += (i+1) * gray[i][j]
#             col_sum += (j+1) * gray[i][j]
#             w = w + gray[i][j]
#     x = int(row_sum/w)
#     y = int(col_sum/w)
#     print(x,y)
#     cv.circle(img1, (x, y), 10, (0, 0, 255), -1)
#     plt.imshow(img1)
#     plt.title('result')
#     plt.show()

def extract_center(image_path):
    img = cv.imread(image_path, 0)
    ret, thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    M = cv.moments(thresh)
    print(f"M:{M}")
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    return (cx, cy)


if __name__ == '__main__':
    # img = cv.imread('../light/Image_20230301205338890.bmp')
    # gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # plt.subplot(121), plt.imshow(gray), plt.title('origin')
    # print(gray.shape)
    # gray1 = gray[400:,1500:1700]
    # plt.subplot(122), plt.imshow(gray1), plt.title('res')
    #
    # plt.show()


    img = cv.imread('../3D/lightplanedemo/light/Image_20230301205338890.bmp')
    cx,cy = extract_center('../3D/lightplanedemo/light/Image_20230301205338890.bmp')
    print(cx,cy)
    cv.circle(img,(cx,cy),10,(0,0,255),-1)
    plt.imshow(img[::-1])
    plt.title('result')
    plt.show()
