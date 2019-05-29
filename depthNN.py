import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Conv2DTranspose, Dropout, Reshape, Activation, ZeroPadding2D, MaxPooling2D, Cropping2D
from keras.optimizers import SGD,Adam
import matplotlib.pyplot as plt

import os

import numpy as np

import tensorflow as tf
print(tf.test.is_built_with_cuda())

data_generator = ImageDataGenerator(validation_split=0.2,
                               rescale=1. / 255)

train_set_size=250382
val_set_size=62595
batch_size=16

def generator_inout(generator, dirIn, dirOut, batch_size, img_height, img_width, subset):
    genI = generator.flow_from_directory(dirIn, subset=subset,
                                          target_size=(img_height, img_width),
                                          class_mode='categorical',
                                          batch_size=batch_size,
                                          shuffle=False)

    genO = generator.flow_from_directory(dirOut, subset=subset,
                                          target_size=(img_height, img_width),
                                          color_mode="grayscale",
                                          class_mode='categorical',
                                          batch_size=batch_size,
                                          shuffle=False)
    while True:
        X = genI.next()
        Y = genO.next()
        yield X[0], Y[0]


training_generator = generator_inout(data_generator,'data/input2','data/output2',batch_size,144,256,'training')
validation_generator = generator_inout(data_generator,'data/input2','data/output2',batch_size,144,256,'validation')



model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(144, 256, 3)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.1))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.1))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))


model.add(ZeroPadding2D(((1, 1), (1, 1))))
model.add(Conv2DTranspose(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D(((1, 1), (1, 1))))
model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu'))
model.add(ZeroPadding2D(((1, 1), (1, 1))))
model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu'))
model.add(Conv2D(1, (3, 3), activation='relu'))
model.add(Cropping2D(cropping=((0, 1), (0, 1))))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.8, nesterov=True)
model.compile(loss='mean_squared_error',
              optimizer=sgd,
              metrics=['accuracy'])

print(model.summary())
mycallback = keras.callbacks.ModelCheckpoint(filepath = 'models/mymodel8_{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
csv_logger = keras.callbacks.CSVLogger('mymodel8.log',append=True)


if os.path.exists('models/mymodel8.h5'):
    model=keras.models.load_model("models/mymodel8.h5")
else:
    model.fit_generator(
		training_generator,
        steps_per_epoch=train_set_size // batch_size,
        epochs=20,
        validation_data=validation_generator,
        validation_steps=val_set_size // batch_size,
        callbacks=[mycallback,csv_logger])
    model.save('models/mymodel8.h5')


