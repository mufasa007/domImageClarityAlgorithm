from numpy import *


class domAlgorithm:
    place = 'China'

    # 初始化方法
    def __init__(self, frame, k):  # 初始化方法
        self.x = frame.shape[0]
        self.y = frame.shape[1]
        self.z = frame.shape[2]
        self.k = k
        kTotal = 2 * k + 1

        self.M[:, :, :, 0] = mat(ones((self.x, self.y, 2 * k + 1))) * frame[:, :, 0]
        self.M[:, :, :, 1] = mat(ones((self.x, self.y, 2 * k + 1))) * frame[:, :, 1]
        self.M[:, :, :, 2] = mat(ones((self.x, self.y, 2 * k + 1))) * frame[:, :, 2]

    # 静态方法


def deal(self, frame, iloop):
    if (iloop % 2 == 0):
        self.M[:, :, 0, 0] = frame[:, :, 0]
        self.M[:, :, 0, 1] = frame[:, :, 1]
        self.M[:, :, 0, 2] = frame[:, :, 2]
    else:
        self.M[:, :, 2 * self.kk + 1, 0] = frame[:, :, 0]
        self.M[:, :, 2 * self.kk + 1, 1] = frame[:, :, 1]
        self.M[:, :, 2 * self.kk + 1, 2] = frame[:, :, 2]

    for z in range(0, 3):
        for x in range(0, self.x):
            for y in range(0, self.y):
                self.M[x, y, :, z] = np.sort(self.M[x, y, :, z])

    return self.M[:,:,self.k,:]


# 类方法
@classmethod
def fun2(arr):
    np.sort(arr)


# 方法
def eat(self):
    print("eating...")
