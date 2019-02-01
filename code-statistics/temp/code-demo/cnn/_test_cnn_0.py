""" 
加载图像数据的函数,dataset_path即图像olivettifaces的路径 
加载olivettifaces后，划分为train_data,valid_data,test_data三个数据集 
函数返回train_data,valid_data,test_data以及对应的label 
"""
import numpy
import theano
import theano.tensor as T
from PIL import Image

dataset_path = "./image/faces.gif"


def load_data(dataset_path="./image/faces.gif"):
    img = Image.open(dataset_path)
    img_ndarray = numpy.asarray(img, dtype="float64") / 256
    faces = numpy.empty((400, 2679))
    for row in range(20):
        for column in range(20):
            faces[row * 20 + column] = numpy.ndarray.flatten(
                img_ndarray[row * 57 : (row + 1) * 57, column * 47 : (column + 1) * 47]
            )

    label = numpy.empty(400)
    for i in range(40):
        label[i * 10 : i * 10 + 10] = i
        label = label.astype(numpy.int)

    # 分成训练集、验证集、测试集，大小如下
    train_data = numpy.empty((320, 2679))
    train_label = numpy.empty(320)
    valid_data = numpy.empty((40, 2679))
    valid_label = numpy.empty(40)
    test_data = numpy.empty((40, 2679))
    test_label = numpy.empty(40)

    for i in range(40):
        train_data[i * 8 : i * 8 + 8] = faces[i * 10 : i * 10 + 8]
        train_label[i * 8 : i * 8 + 8] = label[i * 10 : i * 10 + 8]
        valid_data[i] = faces[i * 10 + 8]
        valid_label[i] = label[i * 10 + 8]
        test_data[i] = faces[i * 10 + 9]
        test_label[i] = label[i * 10 + 9]

    # 将数据集定义成shared类型，才能将数据复制进GPU，利用GPU加速程序。
    def shared_dataset(data_x, data_y, borrow=True):
        shared_x = theano.shared(
            numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow
        )
        shared_y = theano.shared(
            numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow
        )
        return shared_x, T.cast(shared_y, "int32")

    train_set_x, train_set_y = shared_dataset(train_data, train_label)
    valid_set_x, valid_set_y = shared_dataset(valid_data, valid_label)
    test_set_x, test_set_y = shared_dataset(test_data, test_label)
    rval = [
        (train_set_x, train_set_y),
        (valid_set_x, valid_set_y),
        (test_set_x, test_set_y),
    ]
    return rval


if __name__ == "__main__":
    res = load_data()
    print(len(res))
    print(res[0])
    print(res[1])
    print(res[2])
