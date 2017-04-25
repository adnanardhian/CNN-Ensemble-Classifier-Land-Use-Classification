from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils
import csv
import pandas as pd

INDEXS = []
SCORES = []

# parameter
BATCH_SIZE = 21
NB_CLASSES = 21
NB_EPOCH = 40
VGG_LAYER = 16 # 16 or 19

# Get label ready for Neural Network
def getLabel(label_path):
  labels = pd.read_csv(label_path)
  temp = labels['class'].values
  results = []
  for i in range(0,len(temp)):
    results.append([temp[i]])
  return np.array(results)

# write the result to external (.csv) file
def write_csv(fname,jum,indexn,namelist):
    for j in range(0,jum):
        with open(fname+'.csv','ab') as csvfile:
            ciswriter = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_MINIMAL)
            ciswriter.writerow([indexn[j],namelist[j]])

Y = getLabel('label.csv')
X = np.load('Input_NN_VGG'+str(VGG_LAYER)+'.npy') #Assuming the preprocess is done, so the file is ready (both 16 and 19)

SPLIT_IDX = len(Y)*9/10		# Get 90% for training, and 10% for testing
(X_train, y_train), (X_test, y_test) = (X[:SPLIT_IDX],Y[:SPLIT_IDX]), (X[SPLIT_IDX:], Y[SPLIT_IDX:])

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

for i in range(0,2): # 2 iterations will be done, and 2 weights will be saved (customizable)
	model = Sequential()
	model.add(Flatten(input_shape=(512,8,8))) #Shape of output from exctractor 
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(NB_CLASSES))
	model.add(Activation('softmax'))

	weight_init = model.get_weights()
	model.set_weights(weight_init)
	
  # Let's train the model using RMSprop
	rmsprop = RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
	model.compile(loss='categorical_crossentropy',
              optimizer=rmsprop,
              metrics=['accuracy'])

	model.fit(X_train, Y_train,
          batch_size=BATCH_SIZE,
          nb_epoch=NB_EPOCH,
          validation_data=(X_test, Y_test),
          shuffle=True)

	model.save_weights('WeightFile/WeightVGG'+str(VGG_LAYER)+'_'+str(i+1)+'.h5')  # WeightFile is the folder that weight after trained will be saved inside, make sure it is already exist
	score = model.evaluate(X_test, Y_test, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])
	SCORES.append(score[1])
	INDEXS.append(i+1)

write_csv('VGG'+str(VGG_LAYER),len(SCORES),INDEXS,SCORES)
