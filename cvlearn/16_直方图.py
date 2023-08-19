"""
@Time ： 2023/5/18 20:51
@Author ： 佟星
"""
import matplotlib.pyplot as plt
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')

# 获取图像直方图
def get_hist():
    img = cv.imread("img_2.png",0)
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    print(hist)
    thresh,result = cv.threshold(img,80,1,cv.THRESH_BINARY)
    # 显示三幅图像
    fig = plt.figure(figsize=(15, 8))
    ax1, ax2, ax3 = fig.subplots(1, 3, sharey=True)
    ax1.imshow(img,cmap='gray')
    ax1.set_title('horizontal')
    ax2.plot(hist)
    ax2.set_title('vertical')
    ax3.imshow(result,cmap='gray')
    ax3.set_title('result')
    fig.suptitle('show', fontsize=20)
    plt.show()

get_hist()