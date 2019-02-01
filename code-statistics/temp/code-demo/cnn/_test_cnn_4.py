import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Convolution2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    MaxPooling2D,
)
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils

from __init__ import MODEL_NAME, RESULT_SAVING_PATH, load_data

model = Sequential()
epochs = 5
batch_size = 32
# x_train=y_train=x_test=y_test=None
train_batch = validation_batch = None


def initial_cnn_model():
    global model
    model.add(
        Conv2D(
            filters=16,
            kernel_size=2,
            padding="valid",
            activation="relu",
            input_shape=(358, 413, 3),
        )
    )
    # model.add(Dense(413,3,activation="relu",input_shape=(358)))
    # model.add(Dense(5))
    # model.add(MaxPooling2D(pool_size=1))
    # model.add(Conv2D(filters=32, kernel_size=2, padding='valid', activation='relu'))
    # model.add(MaxPooling2D(pool_size=1))
    # model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='valid'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(BatchNormalization())#  防止梯度消失
    # model.add(Activation('relu',i))
    model.summary()


def compile():
    global model
    model.compile(optimizer="rmsprop", loss="mean_squared_error", metrics=["accuracy"])
    pass


def load_data_locl():
    data, label = load_data()
    return data, label


def train_data(data, validation):
    global model
    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd)
    # print(str(data));print(str(label))
    # data = tf.reshape(data,[-1,358, 413, 3])
    model.fit(
        data, validation, epochs=10, shuffle=True, verbose=1, validation_split=0.2
    )


def save_model():
    global model
    model.save(RESULT_SAVING_PATH + "/" + MODEL_NAME)


def get_model_loss():
    global model, x_test, y_test, batch_size
    return model.evaluate(x_test, y_test, batch_size=batch_size)


def predict(test_modle):
    global model
    model.predict(test_modle, batch_size=128)


if __name__ == "__main__":
    print("start init cnn model")
    x, y = load_data_locl()
    print("load data finished!!")
    initial_cnn_model()
    print("initial cnn model finished!")
    train_data(x, y)
    print("train data finieshed")
    save_model()
    print("save model finished")
    print(get_model_loss())


### 不要修改下方代码

# checkpointer = ModelCheckpoint(filepath='results/weights.best.from_scratch.hdf5',
#                                verbose=1, save_best_only=True)

# model.fit(train_tensors, train_targets,
#           validation_data=(valid_tensors, valid_targets),
#           epochs=epochs, batch_size=20, callbacks=[checkpointer], verbose=1)


# #预测
# dog_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]
# ​
# # 报告测试准确率
# test_accuracy = 100*np.sum(np.array(dog_breed_predictions)==np.argmax(test_targets, axis=1))/len(dog_breed_predictions)
