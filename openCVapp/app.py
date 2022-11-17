import cv2
import time
# from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
import random
from gevent import spawn, joinall, monkey;
from concurrent.futures import ThreadPoolExecutor

monkey.patch_all()

# 全局变量 #

# 矩阵时间维度
k = 10

# 循环数据
iloop = 0

# 图片矩阵size
x = 640
y = 480
z = 3

matrix = 1
theard_pool = ThreadPoolExecutor(max_workers=16)

def deal(frame):
    global iloop, matrix,theard_pool
    if (iloop % 2 == 0):
        matrix[:, :, 0, 0] = frame[:, :, 0]
        matrix[:, :, 1, 0] = frame[:, :, 1]
        matrix[:, :, 2, 0] = frame[:, :, 2]
    else:
        matrix[:, :, 0, 2 * k] = frame[:, :, 0]
        matrix[:, :, 0, 2 * k] = frame[:, :, 1]
        matrix[:, :, 0, 2 * k] = frame[:, :, 2]

    a = time.time()
    result = np.empty((x,y,z))
    for i in range(0, x):  # 640
        for j in range(0, y):  # 480
            for m in range(0, z):  # 3
                # single_bubble_sort(iloop, matrix[i, j, m, :].copy())

                theard_pool.submit(single_bubble_sort,iloop, matrix[i, j, m, :].copy())
                # result[i,j,m] = theard_pool.submit(single_bubble_sort,matrix[i, j, m, :].copy())
                # single_bubble_sort(matrix[i, j, m, :])
                # joinall(spawn(single_bubble_sort, matrix[i, j, m, :]))

    b = time.time()
    print("处理耗时：%s", b - a)

def single_bubble_sort(i,arr):
    if (i % 2 == 0):
        for i in range(0, arr.size - 1):
            if (arr[i] < arr[i + 1]):
                return arr
            else:
                mide = arr[i]
                arr[i] = arr[i + 1]
                arr[i + 1] = mide
    else:
        for i in range(arr.size - 1, 0):
            if (arr[i] < arr[i + 1]):
                return arr
            else:
                mide = arr[i]
                arr[i] = arr[i + 1]
                arr[i + 1] = mide
    return arr

def video_demo():
    global iloop, x, y, z, k, matrix

    capture = cv2.VideoCapture(0)  # 0为电脑内置摄像头
    resize = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(resize)
    print(time.strftime('%Y-%m-%d %H:%M:%S'))

    # 数值初始化
    ret, frame = capture.read()
    x = frame.shape[0]
    y = frame.shape[1]
    z = frame.shape[2]

    matrix = np.expand_dims(frame, axis=3).repeat(21, axis=3)

    while (True):
        ret, frame = capture.read()  # 摄像头读取,ret为是否成功打开摄像头,true,false。 frame为视频的每一帧图像

        # 原始图片
        # origin = cv2.flip(frame, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示。
        # frame = origin.copy()

        z0 = np.random.rand(x, y) - 1

        # 模糊处理
        frame[:, :, 0] = frame[:, :, 0] + frame[:, :, 0] * z0 * 0.3
        frame[:, :, 1] = frame[:, :, 1] + frame[:, :, 1] * z0 * 0.3
        frame[:, :, 2] = frame[:, :, 2] + frame[:, :, 2] * z0 * 0.3

        frame[np.where(frame > 255)] = 255
        frame[np.where(frame < 0)] = 0

        # dom算法结果
        deal(frame)
        iloop = iloop + 1

        # 图片显示
        cv2.namedWindow('multi', 0)
        cv2.imshow('multi', matrix[:, :, :, k])
        print(iloop)
        c = cv2.waitKey(50)
        if c == 27:
            break


video_demo()
cv2.destroyAllWindows()
