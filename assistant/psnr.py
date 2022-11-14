from PIL import Image
import numpy as np

img1 = np.array(Image.open('original.jpg')).astype(np.float64)
img2 = np.array(Image.open('compress.jpg')).astype(np.float64)


def psnr(img1, img2):
    mse = np.mean((img1-img2)**2)
    if mse == 0:
        return float('inf')
    else:
        return 20*np.log10(255/np.sqrt(mse))


if __name__ == "__main__":
    print(psnr(img1, img2))
