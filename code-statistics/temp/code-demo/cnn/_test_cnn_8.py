from __init__ import (
    DOG_JINMAO_DIR,
    DOG_PIC_HEIGHT,
    DOG_PIC_WITDH,
    DOG_SUMU_DIR,
    test_jinmao,
    write_log,
)
from generate_model import DogsKindModl, generate_trans_data, predict_apicture


def test_model():
    dog_dir = [DOG_SUMU_DIR, DOG_JINMAO_DIR]
    pic_width = DOG_PIC_WITDH
    pic_height = DOG_PIC_HEIGHT
    trains, labels = generate_trans_data(dog_dir, pic_width, pic_height)
    print(trains.shape, labels.shape)
    write_log(str(labels), file="test_8.log")
    model = DogsKindModl(pic_width, pic_height)
    model.compile_model()
    model.fit_model(trains, labels)
    model.save_model()


def test_predict_a_picture():
    pic_path = "./image/temp/sumu_td.jpg"
    pic_path = test_jinmao
    model_path = "./results/hdf5mdl/test.h5"
    predict_apicture(pic_path, model_path, DOG_PIC_WITDH, DOG_PIC_HEIGHT)


def test_predict_man_face():
    from keras.applications import ResNet50

    model = ResNet50(weights="imagenet")


if __name__ == "__main__":
    # test_model()
    test_predict_a_picture()
