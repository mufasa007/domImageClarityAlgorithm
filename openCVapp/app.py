import cv2
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import random

import domAlgorithm


def show_multi_imgs(scale, imglist, order=None, border=10, border_color=(255, 255, 0)):
    """
    :param scale: float 原图缩放的尺度
    :param imglist: list 待显示的图像序列
    :param order: list or tuple 显示顺序 行×列
    :param border: int 图像间隔距离
    :param border_color: tuple 间隔区域颜色
    :return: 返回拼接好的numpy数组
    """
    if order is None:
        order = [1, len(imglist)]
    allimgs = imglist.copy()
    ws , hs = [], []
    for i, img in enumerate(allimgs):
        if np.ndim(img) == 2:
            allimgs[i] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        allimgs[i] = cv2.resize(img, dsize=(0, 0), fx=scale, fy=scale)
        ws.append(allimgs[i].shape[1])
        hs.append(allimgs[i].shape[0])
    w = max(ws)
    h = max(hs)
    # 将待显示图片拼接起来
    sub = int(order[0] * order[1] - len(imglist))
    # 判断输入的显示格式与待显示图像数量的大小关系
    if sub > 0:
        for s in range(sub):
            allimgs.append(np.zeros_like(allimgs[0]))
    elif sub < 0:
        allimgs = allimgs[:sub]
    imgblank = np.zeros(((h+border) * order[0], (w+border) * order[1], 3)) + border_color
    imgblank = imgblank.astype(np.uint8)
    for i in range(order[0]):
        for j in range(order[1]):
            imgblank[(i * h + i*border):((i + 1) * h+i*border), (j * w + j*border):((j + 1) * w + j*border), :] = allimgs[i * order[1] + j]
    return imgblank

    # dom有序矩阵图像清晰算法


def video_demo():
    a = time.time()
    print(a)
    capture = cv2.VideoCapture(0)  # 0为电脑内置摄像头
    # 更改分辨率大小和fps大小
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    capture.set(cv2.CAP_PROP_FPS, 70)
    resize = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(resize)
    # 记录调用时长
    print(time.time() - a)
    print(time.strftime('%Y-%m-%d %H:%M:%S'))
    # domAlgorithm.domAlgorithm

    fps = capture.get(cv2.CAP_PROP_FPS)
    ret, frame = capture.read()  # 摄像头读取,ret为是否成功打开摄像头,true,false。 frame为视频的每一帧图像
    origin = cv2.flip(frame, 1)

    dom = domAlgorithm.domAlgorithm(origin,10)
    i = 0

    while (True):
        fps = capture.get(cv2.CAP_PROP_FPS)
        # print(fps)
        ret, frame = capture.read()  # 摄像头读取,ret为是否成功打开摄像头,true,false。 frame为视频的每一帧图像

        # 原始图片
        origin = cv2.flip(frame, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示。
        frame = origin.copy()

        x = frame.shape[0]
        y = frame.shape[1]
        z = frame.shape[2]


        # 模糊处理
        z0 = np.random.rand(x,y)-1
        z1 = np.random.rand(x,y)-1
        z2 = np.random.rand(x,y)-1



        frame[:, :, 0] = frame[:,:,0] +  frame[:,:,0] * z0 * 0.3
        frame[:, :, 1] = frame[:,:,1] +  frame[:,:,1] * z1* 0.3
        frame[:, :, 2] = frame[:,:,2] +  frame[:,:,2] * z2* 0.3


        # dom算法结果
        dom_data = dom.deal(frame,i)
        i = i+1

        # data_Laplacian
        # gray_lap = cv2.Laplacian(frame, cv2.CV_16S, ksize=3)
        data_Laplacian = cv2.convertScaleAbs(frame)

        # data_GaussianBlur
        data_Canny = cv2.Canny(frame, 50, 100)

        # data_GaussianBlur
        data_GaussianBlur = cv2.GaussianBlur(frame, (5, 5), 0)

        img = show_multi_imgs(0.9, [origin, frame, dom_data, data_GaussianBlur,  frame, frame], (2, 3))

        print(psnr(frame,data_Laplacian))
        cv2.namedWindow('multi', 0)
        cv2.imshow('multi', img)

        # cv2.imshow("video1", frame)
        # cv2.imshow("data_GaussianBlur", data_GaussianBlur)
        c = cv2.waitKey(50)
        if c == 27:
            break


video_demo()
cv2.destroyAllWindows()
