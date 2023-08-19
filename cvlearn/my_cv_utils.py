# Author : 佟星TONGXING
# Date : 2022/11/24 21:32
# Version: 1.0
import cv2
import matplotlib.pyplot as plt
def cv_show(img):
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('img', 640, 480)
    cv2.imshow("img",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plt_show(img):
    plt.imshow(img[:, :, ::-1])
    plt.title('result')
    plt.show()