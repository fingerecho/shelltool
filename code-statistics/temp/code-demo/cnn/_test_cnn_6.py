import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras

from __init__ import (
    DOG_JINMAO_DIR,
    DOG_JINMAO_SAMPLE_DIR,
    DOG_SUMU_DIR,
    DOG_SUMU_SAMPLE_DIR,
    test_6_MODEL,
    test_jinmao,
    test_sumu,
    write_log,
)

log_file = "_test_6.log"

DOG_PIC_WIDTH = 585
DOG_PIC_HEIGHT = 460

classifier = ["金毛犬", "苏牧犬"]


def generate_model(model_name: str):

    train_size = len(os.listdir(DOG_SUMU_DIR)) + len(os.listdir(DOG_JINMAO_DIR))
    train_fi = len(os.listdir(DOG_JINMAO_DIR))
    # (72,585,460,3)
    trains = np.empty((train_size, DOG_PIC_WIDTH, DOG_PIC_HEIGHT, 3), dtype="float32")
    labels = np.empty((train_size), dtype="int32")
    for ind in range(len(os.listdir(DOG_JINMAO_DIR))):
        img = Image.open(os.path.join(DOG_JINMAO_DIR, os.listdir(DOG_JINMAO_DIR)[ind]))
        arr = np.asarray(img, dtype="float32")
        trains[ind, :, :, :] = arr.reshape(
            DOG_PIC_WIDTH, DOG_PIC_HEIGHT, 3
        )  # (460,585,3)
        labels[ind] = 0

    for ind in range(len(os.listdir(DOG_SUMU_DIR))):
        img = Image.open(
            os.path.join(DOG_SUMU_DIR, os.listdir(DOG_SUMU_DIR)[ind])
        )  # tf.image.decode_jpeg(img,channels=3)
        arr = np.asarray(img, dtype="float32")
        trains[ind + train_fi, :, :, :] = arr.reshape(DOG_PIC_WIDTH, DOG_PIC_HEIGHT, 3)
        labels[ind + train_fi] = 1

        # (74,460,585,3)
    write_log(str(labels), file=log_file)

    model = keras.Sequential(
        [
            keras.layers.Flatten(input_shape=(DOG_PIC_WIDTH, DOG_PIC_HEIGHT, 3)),
            keras.layers.Dense(
                2, activation=tf.nn.relu, input_shape=(DOG_PIC_WIDTH, DOG_PIC_HEIGHT, 3)
            ),
            keras.layers.Dense(2, activation=tf.nn.relu),
            keras.layers.Dense(2, activation=tf.nn.softmax),
        ]
    )

    model.compile(
        optimizer=tf.train.AdamOptimizer(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # expect (585,460,3)
    model.fit(trains, labels, epochs=1, steps_per_epoch=20)
    model.save(test_6_MODEL + "/" + model_name)


# generate_model("1533_02.h5")
def create_test_samples():
    jinmaos = os.listdir(DOG_JINMAO_SAMPLE_DIR)
    sumus = os.listdir(DOG_SUMU_SAMPLE_DIR)
    sp_size = len(jinmaos) + len(sumus)
    sp_fi = len(jinmaos)
    samples = np.empty((sp_size, DOG_PIC_WIDTH, DOG_PIC_HEIGHT, 3))
    labels = np.empty(sp_size)
    for ind in range(len(jinmaos)):
        img = Image.open(os.path.join(DOG_JINMAO_SAMPLE_DIR, jinmaos[ind]))
        arr = np.asarray(img, dtype="float32")
        samples[ind, :, :, :] = arr.reshape(DOG_PIC_WIDTH, DOG_PIC_HEIGHT, 3)
        labels[ind] = 0
    for ind in range(len(sumus)):
        img = Image.open(os.path.join(DOG_SUMU_SAMPLE_DIR, sumus[ind]))
        arr = np.asarray(img, dtype="float32")
        samples[ind + sp_fi, :, :, :] = arr.reshape(DOG_PIC_WIDTH, DOG_PIC_HEIGHT, 3)
        labels[ind] = 1
    return samples, labels


def load_model_and_evaluate(model_name):
    model = keras.models.load_model(os.path.join(test_6_MODEL, model_name))
    test_inputs, test_labels = create_test_samples()
    if model:
        test_loss, test_acc = model.evaluate(test_inputs, test_labels)
        write_log("loss:%s \n acc:%s" % (str(test_loss), str(test_acc)))
    return model


def predict_dog(model, predict_img=test_jinmao):
    img = Image.open(predict_img)
    test = np.asarray(img, dtype="float32")
    sample = np.empty((1, DOG_PIC_WIDTH, DOG_PIC_HEIGHT, 3))
    sample[0, :, :, :] = test.reshape(DOG_PIC_WIDTH, DOG_PIC_HEIGHT, 3)
    predictions = model.predict(sample)
    write_log(str(predictions), file=log_file)


def generate_model_then_predict_model(model_name: str, epochs=2):

    train_size = len(os.listdir(DOG_SUMU_DIR)) + len(os.listdir(DOG_JINMAO_DIR))
    train_fi = len(os.listdir(DOG_JINMAO_DIR))
    trains = np.empty((train_size, DOG_PIC_WIDTH, DOG_PIC_HEIGHT, 3), dtype="float32")
    labels = np.empty((train_size), dtype="int32")

    for ind in range(len(os.listdir(DOG_JINMAO_DIR))):
        img = Image.open(os.path.join(DOG_JINMAO_DIR, os.listdir(DOG_JINMAO_DIR)[ind]))
        arr = np.asarray(img, dtype="float32")
        trains[ind, :, :, :] = arr.reshape(
            DOG_PIC_WIDTH, DOG_PIC_HEIGHT, 3
        )  # (460,585,3)
        labels[ind] = 0

    for ind in range(len(os.listdir(DOG_SUMU_DIR))):
        img = Image.open(
            os.path.join(DOG_SUMU_DIR, os.listdir(DOG_SUMU_DIR)[ind])
        )  # tf.image.decode_jpeg(img,channels=3)
        arr = np.asarray(img, dtype="float32")
        trains[ind + train_fi, :, :, :] = arr.reshape(DOG_PIC_WIDTH, DOG_PIC_HEIGHT, 3)
        labels[ind + train_fi] = 1

    write_log(str(trains.shape), file=log_file)

    model.compile(
        optimizer=tf.train.AdamOptimizer(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # model.fit(trains,epochs=epochs,steps_per_epoch=20)
    model.fit(trains)
    model.save(test_6_MODEL + "/" + model_name)
    # predict_dog(model,test_jinmao)


if __name__ == "__main__":
    generate_model_then_predict_model("1418_02.h5", 1)
