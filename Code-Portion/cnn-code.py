from keras.models import Sequential
from sklearn.externals import joblib
from keras.layers import Convolution2D
from keras.layers.core import Dropout
from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
model = Sequential()
model.add(Convolution2D(64, kernel_size=(3, 3), strides=(1, 1),padding="same",
                input_shape=(64,64,3),  activation='relu', data_format='channels_last'
                 ))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(units =49,activation='softmax'))

model.add(BatchNormalization(axis=1))

model.add(Dropout(0.25))



model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,
                                   zoom_range = 0.5,
                                 horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('/Users/parthsmacbookair/Desktop/MR2/train',
                                                  target_size = (64, 64),
                                                  batch_size = 256,

                                                  class_mode = 'categorical')


test_set = test_datagen.flow_from_directory('/Users/parthsmacbookair/Desktop/MR2/Test',
                                              target_size = (64, 64),

                                                batch_size = 256,

                                                class_mode = 'categorical')


model=model.fit_generator(training_set,

                                           samples_per_epoch = 1925,

                                           nb_epoch = 2,

                                           validation_data = test_set,

                                          nb_val_samples = 623)
# save the model to disk
filename = 'monument_classification.sav'
#joblib.dump(model, filename)
model.save(filename)

'''

# load the model from disk
model = joblib.load(open(filename, 'rb'))


#ploat the map
import matplotlib.pyplot as plt

train_loss=model.history['loss']

train_acc=model.history['acc']

xc=range(50)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)

plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])

plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)

plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])


#for testing



import cv2
import numpy as np
img1 = image.load_img('/home/cloud/Desktop/dog-breed-identification/test/0a0b97441050bba8e733506de4655ea1.jpg')

from keras.models import load_model
from keras.preprocessing import image




images = np.vstack([x])
classes = model.predict_classes(images, batch_size=10)
print (classes)

'''
