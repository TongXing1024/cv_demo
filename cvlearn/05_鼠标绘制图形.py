import numpy as np
import cv2 as cv

img = np.zeros((512,512,3),np.uint8)
#鼠标回调函数
def draw_circle(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDBLCLK:
        cv.circle(img,(x,y),100,(255,0,0),-1)

cv.namedWindow('image')
cv.setMouseCallback('image',draw_circle)
while True:
    cv.imshow('image',img)
    if cv.waitKey(10) & 0xff == 27:
        break
cv.destroyAllWindows()
