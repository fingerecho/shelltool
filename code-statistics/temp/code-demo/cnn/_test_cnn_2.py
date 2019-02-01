
from keras.layers import (Conv2D, Dense, Dropout, Flatten,
                          GlobalAveragePooling2D, MaxPooling2D)
from keras.layers.core import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


​
# model = Sequential()
# ​
# ### TODO
# model.add(Conv2D(filters=16, kernel_size=2, padding='valid', activation='relu',input_shape=(224,224,3)))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Conv2D(filters=32, kernel_size=2, padding='valid', activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(Conv2D(filters=64, kernel_size=2, padding='valid', activation='relu'))
# model.add(MaxPooling2D(pool_size=2))
# model.add(GlobalAveragePooling2D(data_format='channels_last'))
# model.add(Dense(133, activation='softmax'))
# model.summary()
