#!/usr/bin/python3
import os

import numpy as np
from PIL import Image
from tensorflow import keras

from __init__ import test_6_MODEL, test_dir, test_jinmao, test_sumu, write_log

model_name = "1915_01.h5"


def predict_dog(dog_pic_path: str, model_name=model_name):
    img = Image.open(os.path.join(test_dir, dog_pic_path))
    arr = np.asarray(img, dtype="float32").reshape(585, 460, 3)
    data = np.empty((1, 585, 460, 3), dtype="float32")
    data[0, :, :, :] = arr

    model = keras.models.load_model(os.path.join(test_6_MODEL, model_name))
    results = model.predict(data)
    write_log(str(results), file="_test_7.log")


if __name__ == "__main__":
    predict_dog(dog_pic_path="jinmao_01.jpg", model_name=model_name)
