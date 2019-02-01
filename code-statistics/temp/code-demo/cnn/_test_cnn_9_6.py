import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.svm import SVC, SVR

from __init__ import write_log

lf = "_test_cnn_9_6.log"


img_z = Image.open(os.path.join(".", "image", "temp", "zhangxueyou.jpg"))
img_x = Image.open(os.path.join(".", "image", "temp", "xietingfeng.jpg"))
img_zh = Image.open(os.path.join(".", "image", "temp", "zhoujielun.jpg"))
img_li = Image.open(os.path.join(".", "image", "temp", "liming.jpg"))

imgs = [img_z, img_x, img_zh, img_li]
datas = np.empty((len(imgs), 220, 220, 3), dtype="float32")  # 4,220,220,3
la = ["张学友", "谢霆锋", "周杰伦", "黎明"]
labels = np.empty((len(la)), dtype=np.string_)
avr = np.empty((220, 220, 3), dtype="float32")
for ind in range(len(imgs)):
    data_ = np.asarray(imgs[ind], dtype="float32")
    datas[ind, :, :, :] = data_
    avr[:, :, :] += data_

print(datas[0].size, datas[0].shape)
# dataT = datas  # 3 220 220 4

# 余弦 rbf
# linear: （x,x')
# polynomial:   is specified by keyword degree,  by coef0.
# rbf: .  is specified by keyword gamma, must be greater than 0.
# sigmoid (), where  is specified by coef0.

# write_log(str(datas),file=lf)
# write_log(str(labels),file=lf)


# write_log(str(datas.shape),file=lf)
# write_log(str(labels.shape),file=lf)

cf = SVR(kernel="rbf")

print(datas.size, datas.shape)
datas = datas.reshape((4, 220 * 220 * 3))
write_log(str(datas[0]), file=lf)
write_log(str(datas[0].shape), file=lf)

labels = ["1", "2", "3", "4"]
cf.fit(datas, labels)

result = cf.predict(datas[0:])

write_log(str(result), file=lf)

# avr = avr/4
# avr = avr.astype("uint8")
# avr_im = Image.fromarray(avr)
# avr_im.show()
# print(avr.shape)
# print(avr.size)
# write_log(str(avr),file=lf)


# xy = np.zeros((220,220),dtype="int32")

# X = []
# Y = []

# # for line in range(len(data)):
# # 	for cols in range(len(data[line])):
# # 		r = data[line,cols,0]
# # 		g = data[line,cols,1]
# # 		b = data[line,cols,2]
# # 		if  255 - r < 20 and 255 -g < 20  and 255 -b < 20:
# # 			xy[line,cols] =  1
# # 			X.append(line)
# # 			Y.append(cols)


# plt.scatter(xy,xy,color="black")

# plt.show()
