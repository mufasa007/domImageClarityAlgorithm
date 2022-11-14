import cv2


class ImageAlgorithm:

    def GaussianBlur(self,frame):
        data = cv2.GaussianBlur(frame, (5, 5), 0)
        return data

    def laplacian(self,frame):
        gray_lap = cv2.Laplacian(frame, cv2.CV_16S, ksize=3)
        data = cv2.convertScaleAbs(gray_lap)
        return data