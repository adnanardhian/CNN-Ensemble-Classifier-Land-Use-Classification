from __future__ import print_function
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from PIL import Image

VGG_LAYER = 16

# get every pixel value from image, and restored it as a numpy array
def getImageData(im_path, start_idx, end_idx):
  imData = []
  for i in range(start_idx,end_idx):
    im = Image.open(im_path+str(i)+'.tif')
    im = np.array(im)
    r = im[:,:,0]
    g = im[:,:,1]
    b = im[:,:,2]
    imData.append([r,g,b])
  return np.array(imData)

if __name__ == '__main__':
	
  # Choosing the number of layer from VGG 
  if VGG_LAYER==16:
    extractor = VGG16(weights='imagenet', include_top=False)
  elif VGG_LAYER==19:
    extractor = VGG19(weights='imagenet', include_top=False)

  # Get ImageData of 1 subsample (1 image every class)
  # 'IMG' is a folder containing image data
  data1 = getImageData('IMG/',0,21)
  X = extractor.predict(data1, verbose=0)
	
  #iterate for all 2100 images (100 times)
  for i in range(1,100):
	  data2 = getImageData('IMG/',i*21,i*21+21)
	  X2 = extractor.predict(data2, verbose=0)
	  X = np.concatenate((X,X2), axis=0)
	  print(X.shape) # to clarify only (optional)
		
  # save the array to external file that will be used for Neural Network
  np.save('Input_NN_VGG'+str(VGG_LAYER)+'.npy',X)
