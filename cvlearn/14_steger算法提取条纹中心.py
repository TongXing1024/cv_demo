import numpy as np
import cv2

def steger_algorithm(img, scale_factor=1):
    # 图像灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 高斯滤波
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Sobel算子计算x和y方向的梯度
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度的方向和强度
    magnitude, angle = cv2.cartToPolar(sobelx, sobely, angleInDegrees=True)

    # 将梯度方向归一化到0-180度之间
    angle = np.mod(angle, 180)

    # 计算光条纹中心
    center = np.zeros(img.shape[:2], np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if angle[i, j] > 90:
                angle[i, j] = angle[i, j] - 180
            if angle[i, j] < -90:
                angle[i, j] = angle[i, j] + 180
            if angle[i, j] > 45:
                center[i, j] = sobely[i, j] / sobelx[i, j]
            else:
                center[i, j] = sobelx[i, j] / sobely[i, j]

    # 缩放光条纹中心
    if scale_factor != 1:
        center = cv2.resize(center, None, fx=scale_factor, fy=scale_factor)

    return center

if __name__ == '__main__':
    img = cv2.imread("../3D/lightplanedemo/light/Image_20230301205338890.bmp")
    steger_algorithm(img)