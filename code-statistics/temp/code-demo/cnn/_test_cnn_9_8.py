import cv2
import numpy as np

from __init__ import man_test_jpg, write_log
from _test_cnn_9_7 import trim_array

lg = "test_cnn_9_8.log"


"""
把图片变成只有边缘的图像，然后就可以很容易的分辨了，
那么对于这张图边缘信息就是有用的，颜色信息就是没有用的。
而且好的特征应该能够区分纽扣和其它圆形的东西的区别。
图像的梯度去掉了很多不必要的信息(比如不变的背景色)，加重了轮廓。
换句话说，你可以从梯度的图像中轻而易举的发现有个人。

在每个像素点，都有一个幅值(magnitude)和方向，
对于有颜色的图片，会在三个channel上都计算梯度。
那么相应的幅值就是三个channel上最大的幅值，
角度(方向)是最大幅值所对应的角。
方向是像素强度变化方向

https://blog.csdn.net/passball/article/details/82254256
https://blog.csdn.net/wjb820728252/article/details/78395092

"""

image = man_test_jpg
im = cv2.imread(image)
rgb = np.asarray(im)
hog = cv2.HOGDescriptor()
h = hog.compute(im)

gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)  # im 在 x 方向上的梯度
gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)


mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)  # 梯度值 和方向(角度)

write_log(str(mag), file=lg)
write_log(str(angle), file=lg)
write_log(str(mag.shape) + "size:" + str(mag.size), file=lg)
write_log(str(angle.shape) + "size:" + str(angle.size), file=lg)

# write_log(str(rgb.shape))
# write_log(str(h.shape))
# write_log(str(rgb.size))
write_log(str(h.size), file=lg)
