import matplotlib

from my_cv_utils import cv_show
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

#图像的加法
#cv.add函数是饱和加法，而numpy加法是模运算
def image_add():
    # x = np.uint8([250])
    # y = np.uint8([10])
    # print(cv.add(x,y))  #255
    # print(x+y)   #4
    A = cv.imread('A.png',cv.IMREAD_COLOR)
    B = cv.imread('B.png',cv.IMREAD_COLOR)
    C = cv.add(A,B)
    cv_show(C)
    D = A+B
    cv_show(D)

#图像的混合  是加法的一种，与加法不同的是两幅图像的权重不同
def image_fix():
    A = cv.imread('A.png', cv.IMREAD_COLOR)
    B = cv.imread('B.png', cv.IMREAD_COLOR)

    #图像混合
    C = cv.addWeighted(A,0.7,B,0,3,0)
    cv_show(C)

#图像的几何变换  缩放 平移 旋转
#缩放
def image_shrink():
    A = cv.imread('A.png',cv.IMREAD_COLOR)
    row,col = A.shape[:2]
    print(f"原始row = {row},col = {col}")
    A1 = cv.resize(A,(row*2,col*2),interpolation=cv.INTER_CUBIC)  #绝对尺寸
    # A1 = cv.resize(A,None,0.5,0.5,interpolation=cv.INTER_CUBIC)  相对尺寸
    row, col = A1.shape[:2]
    print(f"结束row = {row},col = {col}")
    cv_show(A1)
    cv_show(A)
#平移
def image_translate():
    A = cv.imread('A.png',cv.IMREAD_COLOR)

    #图像平移
    rows,cols = A.shape[:2]
    M = np.float32([[1,0,100],[0,1,50]]) #平移矩阵
    dst = cv.warpAffine(A,M,(cols,rows))

    cv_show(dst)

#旋转
def image_revolve():
    A = cv.imread('A.png', cv.IMREAD_COLOR)
    rows,cols = A.shape[:2]
    #生成旋转矩阵
    M = cv.getRotationMatrix2D((cols/2,rows/2),45,1)
    #进行旋转变换
    dst = cv.warpAffine(A,M,(cols,rows))
    cv_show(dst)

#仿射变换
def image_affine():
    A = cv.imread('A.png', cv.IMREAD_COLOR)
    rows, cols = A.shape[:2]
    #生成变换矩阵
    pts1 = np.float32([[50,50],[200,50],[50,200]])
    pts2 = np.float32([[100,100],[200,50],[50,200]])
    M = cv.getAffineTransform(pts1,pts2)
    dst = cv.warpAffine(A,M,(cols,rows))
    cv_show(dst)

#透视变换
def image_perspective():
    A = cv.imread('A.png', cv.IMREAD_COLOR)
    rows, cols = A.shape[:2]
    #创建变换矩阵
    pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    pts2 = np.float32([[100,145],[300,100],[80,290],[310,300]])
    T = cv.getPerspectiveTransform(pts1,pts2)
    #进行变换
    dst = cv.warpPerspective(A,T,(cols,rows))
    #图像显示
    matplotlib.use("TkAgg")
    fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(10,8),dpi=100)
    axes[0].imshow(A[:,:,::-1])
    axes[0].set_title("原图")
    axes[1].imshow(dst[:,:,::-1])
    axes[1].set_title("透视图")
    plt.show()

#图像金字塔
def image_tower():
    img = cv.imread('food.jpg', cv.IMREAD_COLOR)
    up = cv.pyrUp(img)
    down = cv.pyrDown(img)
    matplotlib.use("TkAgg")
    plt.imshow(down[:,:,::-1])
    plt.show()

if __name__ == '__main__':
    #图像加法
    # image_add()

    #图像混合
    # image_fix()

    #缩放
    # image_shrink()

    #平移
    # image_translate()

    #旋转
    #image_revolve()

    #仿射变换
    # image_affine()

    #透视变换
    # image_perspective()

    #金字塔
    image_tower()