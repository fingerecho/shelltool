import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from __init__ import (
    haar_cascade_dir,
    haar_frontalface_default_xml,
    harr_cascade_eye_xml,
    man_test_jpg,
    write_log,
)

lg = "_test_cnn_9_7.log"


def opencv_CascadeClassify():
    """
		使用OpenCV的人脸检测器进行人脸的初步检测，
		使用框架训练CNN网络进行人脸的二分类判定，
		将两部分合在一起完成人脸检测。
		此环节需注意根据应用场景调整参数，
		做到性能与召回率的平衡。
		
		args:
		returns:
			(x,y,w,h) type: tuple
			x,y 返回人脸矩形的左上角坐标
			w,y 返回矩形的宽和高

	"""

    for xml in os.listdir(haar_cascade_dir):
        xml = os.path.join(haar_cascade_dir, xml)
        # write_log(str(xml),file=lg)
        face_cascade = cv.CascadeClassifier(xml)
        # eye_cascade = cv.CascadeClassifier(harr_cascade_eye_xml)
        img = cv.imread(man_test_jpg)
        print(type(img))
        print(str(img.shape))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        print(type(gray))
        write_log(str(type(img)), file=lg)
        write_log(str(img), file=lg)
        write_log(str(type(gray)), file=lg)
        write_log(str(gray), file=lg)
        cv.waitKey(0)

        faces = face_cascade.detectMultiScale(
            img,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 30),
            # flags=cv.cv.CV_HAAR_SCALE_IMAGE
        )
        print("length of faces is %d" % (len(faces)))
        if len(faces) == 0:
            # write_log("failed this file %s"%(xml),file=lg)
            continue
        else:
            # write_log("success this file %s"%(xml),file=lg)
            for x, y, w, h in faces:
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
                write_log(str(w) + str(h), file=lg)
            return faces
            # cv.imshow("myself.com",img)
            # cv.waitKey(0)
            # write_log(str(faces),file=lg)
            break


def trim_array(x, y, w, h):
    """
	args:
		x,y 返回人脸矩形的左上角坐标
		w,y 返回矩形的宽和高
	returns:
		data: 
			type np.ndarray
			裁剪到的 numpy target area 
	"""

    x_ = x + w
    y_ = y + h
    print(x, y, x_, y_)
    img = Image.open(man_test_jpg)
    data = np.asarray(img)
    print(data.shape)
    data = data[y:y_, x:x_, :]  # 191 191 3  需要转换一下 transponse
    # data = data.astype("uint8");print(data.shape)
    # img_t = Image.fromarray(data)
    # img_t.show()
    return data


def hog_feature_process():
    """
	HOG特征提取
	"""
    return None


if __name__ == "__main__":
    results = opencv_CascadeClassify()
    x, y, w, h = results[0]
    print(x, y, w, h)
    # a(x,y,w,h)

    # write_log(str(results[0]),file=lg)


# write_log(str(type(faces)),file=lg)
# print(faces)
# print("lenght of faces: %d"%(len(faces)))
# write_log(str(faces),file=lg)
# for (x,y,w,h) in faces:
#     cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#     roi_gray = gray[y:y+h, x:x+w]
#     roi_color = img[y:y+h, x:x+w]
#     eyes = eye_cascade.detectMultiScale(roi_gray)
#     for (ex,ey,ew,eh) in eyes:
#         cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

# cv.imshow("myself.com",gray)
# cv.waitKey(0)
