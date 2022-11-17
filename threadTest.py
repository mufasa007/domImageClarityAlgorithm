import threading
from datetime import datetime
import numpy as np


def thread_func():  # 线程函数
    print('我是一个线程函数', datetime.now())


def many_thread():
    threads = []
    for _ in range(10):  # 循环创建10个线程
        t = threading.Thread(target=thread_func)
        threads.append(t)
    for t in threads:  # 循环启动10个线程
        t.start()


if __name__ == '__main__':
    a = np.ones((2, 2, 2))
    a[:, :, 0] = np.zeros((2, 2))
    print(a)
    many_thread()
