import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from sklearn.utils.multiclass import unique_labels
import itertools
import matplotlib.pyplot as plt

fig = plt.gcf()

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, cm[i,j],
      horizontalalignment = 'center',
      color = "white" if cm[i,j] > thresh else "black")

    fig.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted label')

def pred(model,Xtest,ytest):
    
    #m_test = Xtest.shape[0]    
    #Xtest = np.reshape(Xtest,(m_test,n_timesteps,n_features))
    y_predicted = model.predict_classes(Xtest, verbose = 2)
    y_actual = ytest
    
    cm = confusion_matrix(y_actual, y_predicted)
    cm_labels = ['Undamaged','Damaged']
    
    return plot_confusion_matrix(cm, classes=cm_labels, title='Confusion matrix'), print(classification_report(y_actual, y_predicted, target_names = cm_labels))

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os

def load_labels(m):
	# initialize the list of column names in the CSV file and then
	# load it using Pandas
	dfUD = pd.DataFrame(np.zeros((m,1), dtype=int))
	dfD = pd.DataFrame(np.ones((m,1), dtype=int))
	# return the data frame
	return dfUD, dfD

def load_imagesUD(dfUD,pathUD):
    imagesUD = []
    for i in dfUD.index.values:
        baseUD = os.path.sep.join([pathUD, "D{}.png".format(i + 1)])
        #print(base)
        image = cv2.imread(baseUD) # read the path using opencv
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (150, 150))
        #plt.imshow(image) # use matplotlib to plot the image
        #image = image[:,:,np.newaxis] #This is convert (600,600) --> (600,600,1)
        imagesUD.append(image) 
    return np.array(imagesUD)

def load_imagesD(dfD,pathD):
    imagesD = []
    for i in dfD.index.values:
        baseD = os.path.sep.join([pathD, "D{}.png".format(i + 1)])
        #print(base)
        image = cv2.imread(baseD) # read the path using opencv
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (150, 150))
        #plt.imshow(image) # use matplotlib to plot the image
        #image = image[:,:,np.newaxis] #This is convert (600,600) --> (600,600,1)
        imagesD.append(image) 
    return np.array(imagesD)

import matplotlib.pyplot as plt
import random
images = []
path = 'D:/SDE/2_SDEcompositeDL/CollectData/3_CWT/UD'
for i in range(2):
    print(i)
    base = os.path.sep.join([path, "D{}.png".format(i + 1)])
    print(base)
    image0 = cv2.imread(base) # read the path using opencv #,cv2.IMREAD_GRAYSCALE
    image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
    image0 = cv2.resize(image0, (150, 150))
    plt.imshow(image0) # use matplotlib to plot the image
    #image = image[:,:,np.newaxis] #This is convert (600,600) --> (600,600,1)
    images.append(image0)

np.array(images).shape

# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model

def create_cnn(width, height, depth, filters=(16, 32, 64)):
	# initialize the input shape and channel dimension, assuming
	# TensorFlow/channels-last ordering
	inputShape = (height, width, depth)
	chanDim = -1

	# define the model input
	inputs = Input(shape=inputShape)

	# loop over the number of filters
	for (i, f) in enumerate(filters):
		# if this is the first CONV layer then set the input
		# appropriately
		if i == 0:
			x = inputs

		# CONV => RELU => BN => POOL
		x = Conv2D(f, (3, 3), padding="same")(x)
		x = Activation("relu")(x)
		x = BatchNormalization(axis=chanDim)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

	# flatten the volume, then FC => RELU => BN => DROPOUT
	x = Flatten()(x)

	# construct the CNN
	model = Model(inputs, x)

	# return the CNN
	return model

# import the necessary packages
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import concatenate
import numpy as np
import argparse
import locale
import os

# construct the path to the input .txt file that contains information
# on each house in the dataset and then load the dataset
pathD = 'D:/SDE/2_SDEcompositeDL/CollectData/3_CWT/D'
pathUD = 'D:/SDE/2_SDEcompositeDL/CollectData/3_CWT/UD'

[dfUD,dfD] = load_labels(2500)
df = np.concatenate([dfUD, dfD], axis=0)
df.shape

print("[INFO] loading UD images...")
imagesUD = load_imagesUD(dfUD, pathUD)
imagesUD = imagesUD / 255.0

print("[INFO] loading D images...")
imagesD = load_imagesD(dfD, pathD)
imagesD = imagesD / 255.0

images = np.concatenate([imagesUD, imagesD], axis=0)
images.shape

split = train_test_split(df, images, test_size=0.25, random_state=42)
(trainAttr, testAttr, trainImages, testImages) = split

trainY = trainAttr
testY = testAttr
print(trainImages.shape)
print(testImages.shape)
print(trainY.shape)
print(testY.shape)

# create the MLP and CNN models
cnn2d = create_cnn(150,150,3)

# create the input to our final set of layers as the *output* of both
Input = cnn2d.output

x = Dense(16, activation="relu")(Input)
x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=cnn2d.input, outputs=x)

opt = Adam(lr=1e-5)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics = ['acc'])

# train the model
print("[INFO] training model...")
history = model.fit(trainImages,trainY,validation_data=(testImages,testY), epochs=250, batch_size=16, verbose = 1)

# Summarize history for loss = BCE
import matplotlib.pyplot as plt
plt.figure(figsize=(14,8))
plt.plot(history.history['loss'],'-o')
plt.plot(history.history['val_loss'],'-s')
plt.title('Loss curve for 2DCNN',fontsize=24)
plt.ylabel('Binary cross entropy',fontsize=22)
#plt.grid()
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.xlabel('Number of epochs',fontsize=22)
plt.legend(['train', 'test'], loc='upper right',fontsize=22)
plt.axis([70,250,0.25e-6,1e-12])
plt.show()

# Summarize history for acc
import matplotlib.pyplot as plt
plt.figure(figsize=(8,4))
plt.plot(history.history['acc'],'-o')
plt.plot(history.history['val_acc'],'-s')
plt.title('Accuracy from 2DCNN',fontsize=18)
plt.ylabel('Acc',fontsize=18)
#plt.grid()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Number of epochs',fontsize=18)
plt.legend(['train', 'test'], loc='upper right',fontsize=10)
plt.axis([0,350,0,10000])
plt.show()

# call pred function
pred(model,testImages,testY)
