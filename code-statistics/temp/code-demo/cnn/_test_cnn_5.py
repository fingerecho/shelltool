import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from __init__ import DOG_JINMAO_DIR, DOG_SUMU_DIR, test_jinmao, test_sumu, write_log

log_file = "./_test_5.log"


class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

fashion_mnist = keras.datasets.fashion_mnist
(x, y), (x_t, y_t) = fashion_mnist.load_data()
x = x / 255.0
x_t = x_t / 255.0

write_log(str(x), file="_test_5.log")
write_log(str(y), file="_test_5.log")
write_log(str(x_t), file="_test_5.log")

model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax),
    ]
)

model.compile(
    optimizer=tf.train.AdamOptimizer(),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(x, y, epochs=1)
test_loss, test_acc = model.evaluate(x_t, y_t)
write_log("loss:%s acc:%s" % (test_loss, test_acc), file=log_file)
pridictions = model.predict(x_t)
max_ = np.argmax(pridictions[1])
write_log("most possible:%s" % (str(max_)), file=log_file)
write_log("possible variable:\n%s" % (str(pridictions[0])), file=log_file)
