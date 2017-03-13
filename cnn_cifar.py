'''
Train a simple deep CNN on the UC Merced Land Use images dataset.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import pandas as pd
from PIL import Image
import numpy as np

# Get image data as an array that will become the input (X) for the Neural Network
# im_path : Image path (Directories)
# start_idx : start index, all images are renamed with number from 0-2099
# end_idx : end index
def getImageData(im_path, start_idx, end_idx):
  imData = []
  for i in range(start_idx,end_idx):
    im = Image.open(im_path+str(i)+'.jpg')
    im = np.array(im)
    r = im[:,:,0]
    g = im[:,:,1]
    b = im[:,:,2]
    imData.append([r,g,b])
  return np.array(imData)

# Get the label value that will become the input (Y) for the Neural Network
# label_path : csv file consist of labels
def getLabel(label_path):
  labels = pd.read_csv(label_path)
  label_value = labels['class'].values
  list_of_label = []
  for i in range(0,len(label_value)):
    list_of_label.append([label_value[i]])
  return np.array(list_of_label)

batch_size = 21 
nb_classes = 21
nb_epoch = 15

# input image dimensions that has been resized
img_rows, img_cols = 128,128
# The images are RGB.
img_channels = 3

X = getImageData('Data128/',0,2100)
Y = getLabel('label.csv')

split_idx = 2100-210 #Get 90% for train, and 10% for test
# The data, split between train and test sets:
(X_train, y_train), (X_test, y_test) = (X[:split_idx],Y[:split_idx]), (X[split_idx:], Y[split_idx:])

print('X_train shape:', X.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(3,128,128)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))   
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          validation_data=(X_test, Y_test),
          shuffle=True)

model.save_weights('WeightCNN.h5') # the weight after train will be saved and used for the ensemble classifier later  
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
