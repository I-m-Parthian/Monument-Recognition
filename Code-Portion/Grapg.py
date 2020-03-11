# -*- coding: utf-8 -*-
"""
Created on Sun May 12 16:08:32 2019

@author: administrator1
"""
from keras.models import Sequential
from sklearn.externals import joblib
from keras.layers import Convolution2D
from keras.layers.core import Dropout
from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization

filename = 'finalized_model.sav'




# load the model from disk
model = joblib.load(open(filename, 'rb'))


#ploat the map
import matplotlib.pyplot as plt

train_loss=model.history['loss']

train_acc=model.history['acc']

val_loss=model.history['val_loss']

val_acc=model.history['val_acc']

xc=range(25)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])

#plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc,val_acc)

plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
#plt.style.use(['classic'])
