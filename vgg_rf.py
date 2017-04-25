
from __future__ import print_function
import numpy as np
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.utils import np_utils

import pandas as pd

from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

NB_CLASSES = 21
VGG_LAYER = 16 #16 or 19

# Random Forest Parameter
SEED = 7
NUM_TREES = 500
MAX_FEATURES = 3

# Get label ready for Neural Network
def getLabel(label_path):
  labels = pd.read_csv(label_path)
  temp = labels['class'].values
  results = []
  for i in range(0,len(temp)):
    results.append([temp[i]])
  return np.array(results)

X = np.load('Input_NN_VGG'+str(VGG_LAYER)+'.npy') #Assuming the preprocess is done, so the file is ready (both 16 and 19)

Y_RF = []
for i in range(0,2100):
  Y_RF.append(Y[i][0])

SPLIT_IDX = len(Y)*9/10		# Get 90% for training, and 10% for testing
(X_train, y_train), (X_test, y_test) = (X[:SPLIT_IDX],Y[:SPLIT_IDX]), (X[SPLIT_IDX:], Y[SPLIT_IDX:])

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_RF_train = Y_RF[:SPLIT_IDX]
Y_RF_test = Y_RF[SPLIT_IDX:]

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

model = Sequential()
model.add(Flatten(input_shape=(512,8,8)))
model.add(Dense(512, activation='relu',name = 'my512'))
model.add(Dropout(0.5))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.load_weights('WeightFile/WeightVGG'+str(VGG_LAYER)+'_'1.h5') # 1 is the index of the weight file that will be used to predict (customizable) 
  
layer_name = 'my512' 
intermediate_layer_model = Model(input=model.input,
                               output=model.get_layer(layer_name).output)  # Get the result of only from feature extraction

output_VGG_train = intermediate_layer_model.predict(X_train)
output_VGG_test = intermediate_layer_model.predict(X_test)    
      
#Random Forest
kfold = model_selection.KFold(n_splits=10, random_state=SEED)
models = RandomForestClassifier(n_estimators=NUM_TREES, max_features=MAX_FEATURES)
models.fit(output_VGG_train,Y_RF_train)
results = model_selection.cross_val_score(models, output_VGG_test, Y_RF_test, cv=kfold) # Use cross validation
print('RF : ',results.mean())

