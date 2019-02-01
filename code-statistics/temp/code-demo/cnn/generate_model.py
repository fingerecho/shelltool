import os

import keras
import numpy as np
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from PIL import Image

from __init__ import NUM_CLASSES, debug

if debug:
    from __init__ import write_log


def generate_trans_data(dog_pics_dir_li, dog_width, dog_height):
    """
    REFERENCES:
        根据提供的狗图片目录 和 狗图片的宽度和高度 生成训练模型需要的狗 np数组
    ARGS: 
        dog_pics_dir_li: ['./image/dogs/hashiqi','./image/dogs/guibin',....] # list
        dog_width: 585 # integer
        dog_height: 460 # integer
    RETURNS:
        trains:(73,585,460,3) # np.array
        labels:(72,) # np.array
    """
    trains_size = 0
    for di in dog_pics_dir_li:
        trains_size = trains_size + len(os.listdir(di))
    trains = np.empty((trains_size, dog_width, dog_height, 3), dtype="float32")
    labels = np.empty((trains_size), dtype="float32")
    partition = 1.000 / NUM_CLASSES
    # write_log(str(partition),file="test_8.log")
    def create_tl(dog_path, dog_width, dog_height, class_type, length=0, ind_start=0):
        nonlocal trains, labels
        tar_li = os.listdir(dog_path)
        for ind in range(len(tar_li)):
            img = Image.open(os.path.join(dog_path, tar_li[ind]))
            arr = np.asarray(img, dtype="float32")
            trains[ind_start + ind, :, :, :] = arr.reshape(
                dog_width, dog_height, 3
            )  # (460,585,3)
            labels[ind_start + ind] = partition * class_type
            length = length + 1
        return length

    ind_start = 0

    # for di in range(len(dog_pics_dir_li)):
    #    ind_start = create_tl(dog_pics_dir_li[di],dog_width,dog_height,
    #        class_type=di,length=ind_start,ind_start=ind_start)
    return trains, labels


def predict_apicture(pic_path, model_path, dog_width, dog_height):
    """
    ARGS:
        pic_path:预测图片路径
        model_path:模型存储路径
    RETURNS:
        result: 分类标签名
    """
    img = Image.open(pic_path)
    arr = np.asarray(img, dtype="float32")
    data = np.empty((1, dog_width, dog_height, 3), dtype="float32")
    data[0, :, :, :] = arr.reshape(dog_width, dog_height, 3)

    from keras.models import load_model

    model = load_model(model_path)
    results = model.predict(data)
    write_log(str(results), file="test_8.log")
    pass


class DogsKindModl(object):
    """
    initial: sgd # sgd优化器
    args:
        pic_width: 图片宽度
        pic_height: 图片高度
        epochs: 模型fit训练的次数,默认测试值为2
    call_model: 构建模型层， 2D卷积层，激活层， 2D卷积池
    call_model_con:构建全连接层
    compile_model:编译模型，使用初始化的sgd优化器
    fit_model: 训练模型,使用初始化的epochs,从模块外部传入训练数据和标签
    save_model: 序列化训练好的模型
    """

    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    def __init__(self, pic_width, pic_height, activate_func="relu", epochs=2):
        self.pic_width = pic_width
        self.pic_height = pic_height
        self.epochs = epochs
        self.activate_func = activate_func
        self.model = keras.models.Sequential()
        self.pic_shape = (self.pic_width, self.pic_height, 3)
        for ker in [8, 16, 32, 64, 128, 256]:
            self.call_model(ker)
        self.call_model_con(dense_ker=16)
        pass

    def call_model_(self):
        # -1,585,460,3   117
        # kernalsize 32
        #
        self.model.add(Conv2D(32, (5, 4), steps=2))
        # 5 , 4   117  114
        # 32
        # self.model.add(Activation())
        # 5, 4    117 114   32*3
        #
        self.model.add(MaxPooling2D(32, (5, 4)))
        #
        self.model.add(Dropout(32, (5, 4)))

    def call_model(self, kernerlsize):
        self.model.add(
            Conv2D(kernerlsize, (3, 3), padding="same", input_shape=self.pic_shape)
        )
        self.model.add(Activation(self.activate_func))
        self.model.add(Conv2D(kernerlsize, (3, 3)))
        self.model.add(Activation(self.activate_func))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        # self.model.add(Dropout(0.25))
        pass

    def call_model_con(self, dense_ker):
        self.model.add(Flatten())
        self.model.add(Dense(dense_ker))
        self.model.add(Activation(self.activate_func))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(NUM_CLASSES))
        self.model.add(Activation("softmax"))
        pass

    def compile_model(self):
        self.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="adam",  # self.sgd,
            metrics=["accuracy"],
        )
        pass

    def fit_model(self, trians, trians_label):
        self.model.fit(trians, trians_label, epochs=self.epochs)
        pass

    def save_model(self, savingpath="./results/hdf5mdl/test.h5"):
        self.model.save(savingpath)
        pass
