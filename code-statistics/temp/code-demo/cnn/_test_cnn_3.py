def predict_pic_using_imagenet():
    from keras.applications.resnet50 import ResNet50
    from keras.preprocessing import image
    from keras.applications.resnet50 import preprocess_input, decode_predictions
    import numpy as np
    from keras.models import load_model

    # model = ResNet50(weights='imagenet')
    model = load_model("results/hdf5mdl/model_0.h5")
    img_path = "image/Cat/1021.jpg"
    img = image.load_img(img_path, target_size=(358, 413))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    # decode the results into a list of tuples (class, description, probability)
    # (one such list for each sample in the batch)
    print(preds)
    # print('Predicted:', decode_predictions(preds, top=3)[0])
    ## Result like that :
    # Predicted: [('n04589890', 'window_screen', 0.2780997),
    # ('n02123045', 'tabby', 0.15884057),
    # ('n02123394', 'Persian_cat', 0.14964087)]
    #########
    ## the second result like that:
    # Predicted: [('n02979186', 'cassette_player', 0.17041288),
    # ('n04392985', 'tape_player', 0.14401877),
    # ('n03459775', 'grille', 0.084367014)]
    ##############


if __name__ == "__main__":
    predict_pic_using_imagenet()
