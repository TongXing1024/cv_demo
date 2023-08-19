import matplotlib
from my_cv_utils import cv_show
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
matplotlib.use('TkAgg')

def image_match():
    img = cv.imread('images/wulin.jpeg', cv.IMREAD_COLOR)
    template = cv.imread('images/bai.jpeg')
    h,w,l = template.shape
    #模板匹配
    res = cv.matchTemplate(img,template,cv.TM_CCORR)
    #返回图像中最匹配的位置，确定左上角的坐标，并将匹配位置绘制在图像上
    min_val,max_val,min_loc,max_loc = cv.minMaxLoc(res)
    #使用平方差时最小值为最佳匹配位置
    top_left = max_loc
    bottom_right = (top_left[0]+w,top_left[1]+h)
    cv.rectangle(img,top_left,bottom_right,(0,255,0),2)
    #显示图形
    plt.imshow(img[:,:,::-1])
    plt.title('result')
    plt.show()


#霍夫变换
def image_hough_line():
    img = cv.imread('images/rili.jpg', cv.IMREAD_COLOR)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) #转换为灰度图
    #边缘检测
    edges = cv.Canny(gray,50,150)
    #霍夫直线变换
    lines = cv.HoughLines(edges,0.8,np.pi/180,150)
    print(lines)
    #将检测的线绘制在图像上
    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(img,(x1,y1),(x2,y2),(0,255,0))
    # 显示图形
    plt.imshow(img[:, :, ::-1])
    plt.title('result')
    plt.show()

def hough_circle():
    planets = cv.imread('images/watertest.jpg', cv.IMREAD_COLOR)
    gray = cv.cvtColor(planets, cv.COLOR_BGR2GRAY)  # 转换为灰度图
    #进行中值模糊，去噪点
    img = cv.medianBlur(gray,7)
    #霍夫圆检测
    circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,200,param1=100,param2=50,minRadius=0,maxRadius=100)
    print(circles.shape)
    print(f"aaaaa:{circles[0,:][0][0]}")
    for i in circles[0,:]:
        #绘制圆形
        print(i[0])
        cv.circle(planets,(int(i[0]),int(i[1])),int(i[2]),(0,255,0),2)
        #绘制圆心
        cv.circle(planets,(int(i[0]),int(i[1])),2,(0,0,255),3)

    plt.imshow(planets[:, :, ::-1])
    plt.title('result')
    plt.show()
if __name__ == '__main__':
    #模式匹配
    # image_match()
    #霍夫变换 线条检测
    image_hough_line()

    # 霍夫变换 圆检测
    # hough_circle()