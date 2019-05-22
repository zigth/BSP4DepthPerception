import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Conv2DTranspose, Dropout, Reshape, Activation, ZeroPadding2D, MaxPooling2D, Cropping2D
from keras.optimizers import SGD,Adam
import matplotlib.pyplot as plt
#from PIL import Image
import os
import random
#import imageio

# Generate dummy data
import numpy as np

import tensorflow as tf
print(tf.test.is_built_with_cuda())

data_generator = ImageDataGenerator(validation_split=0.2,
                               rescale=1. / 255)

train_set_size=250382
val_set_size=62595
batch_size=16

def generator_training_inout(generator, dirIn, dirOut, batch_size, img_height, img_width):
    genI = generator.flow_from_directory(dirIn, subset='training',
                                          target_size=(img_height, img_width),
                                          class_mode='categorical',
                                          batch_size=batch_size,
                                          shuffle=False)

    genO = generator.flow_from_directory(dirOut, subset='training',
                                          target_size=(img_height, img_width),
                                          color_mode="grayscale",
                                          class_mode='categorical',
                                          batch_size=batch_size,
                                          shuffle=False)
    while True:
        X = genI.next()
        Y = genO.next()
        yield X[0], Y[0]

def generator_validation_inout(generator, dirIn, dirOut, batch_size, img_height, img_width):
    genI = generator.flow_from_directory(dirIn, subset='validation',
                                          target_size=(img_height, img_width),
                                          class_mode='categorical',
                                          batch_size=batch_size,
                                          shuffle=False)

    genO = generator.flow_from_directory(dirOut, subset='validation',
                                          target_size=(img_height, img_width),
                                          color_mode="grayscale",
                                          class_mode='categorical',
                                          batch_size=batch_size,
                                          shuffle=False)
    while True:
        X = genI.next()
        Y = genO.next()
        yield X[0], Y[0]


#training_generator = generator_training_inout(data_generator,'data/input','data/output',batch_size,144,256)
validation_generator = generator_validation_inout(data_generator,'data/input2','data/output2',batch_size,144,256)


if os.path.exists('models/mymodel8_15-0.02.h5'):
    model=keras.models.load_model("models/mymodel8_15-0.02.h5")

#for i in range(random.randint(0,1000)):
#	data=next(validation_generator)

fig = plt.figure(figsize=(12,15))
		
#img_path = "data/input/scene/ims_10002382_42314.png"
#img = image.load_img(img_path, target_size=(144, 256))
#print(type(img))

#x = image.img_to_array(img)/255.
#print(type(x))
#print(x.shape)
#fig.add_subplot(3,1,1)
#plt.imshow(x)
#plt.show()

#x = np.expand_dims(x, axis=0)
#print(x.shape)

#fig.add_subplot(3,1,2)
#img_depth = model.predict(x).reshape(144,256)
#print(img_depth)

#plt.imshow(img_depth,cmap='gray')
#plt.show()

#img_path = "data/output/depth/imd_10002382_42314.png"
#img = image.load_img(img_path, target_size=(144, 256))
#print(type(img))

#x = np.array(image.img_to_array(img)/255.)
#print(type(x))
#print(x.shape)
#fig.add_subplot(3,1,3)
#plt.imshow(x,cmap='jet')
#plt.show()

for i in range(0,4):
    data=next(validation_generator)
    fig.add_subplot(4, 3, i*3+1)
    plt.imshow(data[0][0])
    fig.add_subplot(4, 3, i*3+2)
    plt.imshow(model.predict(data[0])[0].reshape(144,256),cmap='jet')
    fig.add_subplot(4, 3, i * 3 + 3)
    plt.imshow(data[1][0].reshape(144,256),cmap='jet')

plt.show()