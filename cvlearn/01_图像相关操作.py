# Author : 佟星TONGXING
# Date : 2022/11/24 15:08
# Version: 1.0
import matplotlib.pyplot as plt
import cv2
import matplotlib


# 显示图像
def plot_show():
    matplotlib.use('TkAgg')
    img = cv2.imread("food.jpg", cv2.IMREAD_COLOR)
    plt.imshow(img[:,:,::-1])
    plt.show()


# 显示图像
def cv_show(name, img):
    """
    图像的显示
    :param name:
    :param img:
    :return:
    """
    cv2.imshow(name, img)
    cv2.waitKey(0)  # 等待时间,毫秒级,0表示任意键终止
    cv2.destroyAllWindows()  # 关闭所有窗口


# 读取图像
def img_read():
    img = cv2.imread("D:/File/cvimg/food.jpg", cv2.IMREAD_COLOR)
    return img


# 保存图像
def img_write(img):
    cv2.imwrite("D:/File/cvimg/f.jpg", img)


def img_split(img):
    new_food = img[0:50, 0:200]
    cv_show('new_food', new_food)


def img_bgr(img):
    # 只保留R通道
    cur_img = img.copy()
    cur_img[:, :, 0] = 0
    cur_img[:, :, 1] = 0
    cv_show('R', cur_img)


# 读取视频
def video_read(videoname):
    vc = cv2.VideoCapture(videoname)  # cv2.VideoCapture()可以捕获摄像头,用数字来控制不同的设备,如0,1。如果是视频文件,直接指定好路径即可
    # 检查是否正常打开
    if vc.isOpened():
        open, frame = vc.read()  # 第一帧
    else:
        open = False
    while open:
        ret, frame = vc.read()
        if frame is None:
            break
        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.namedWindow('myvideo', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('myvideo', 640, 480)
            cv2.imshow("myvideo", gray)
            if cv2.waitKey(10) & 0xff == 27:
                break
    vc.release()
    cv2.destroyAllWindows()


def video_write():
    cap = cv2.VideoCapture("dog.mp4")
    # 定义编解码器并创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (600, 480))
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 0)

        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(10) & 0xff == 27:
            break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

def test():
    img = cv2.imread("food.jpg", 0)
    cv2.imshow('img', img)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
    elif k == ord('s'):
        cv2.imwrite('food1.jpg', img)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    pass

    # test()
    # 图片相关操作
    # cv_show('img',img_read())
    # img_write(img_read())
    # img_split(img_read())
    # img_bgr(cv2.imread("D:/File/cvimg/food.jpg"))
    plot_show()
    # 视频操作
    #video_read("dog.mp4")
    # video_write()
