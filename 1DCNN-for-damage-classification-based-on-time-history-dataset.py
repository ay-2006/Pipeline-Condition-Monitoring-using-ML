def split1(X, y, mvalid, mtest): # full dataset
    
    # split
    X1, Xtest, y1, ytest = train_test_split(X,y, test_size = mtest, random_state = 42)
    Xtrain, Xvalid, ytrain, yvalid = train_test_split(X1,y1, test_size = mvalid, random_state = 42)
    
    return [Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest];

def split2(Xn, yn, mvalid, mtest): # for noise + full dataset
    
    import random
    
    # split
    X1, Xtest, y1, ytest = train_test_split(Xn,yn, test_size = mtest, random_state = 42)
    Xtrain, Xvalid, ytrain, yvalid = train_test_split(X1,y1, test_size = mvalid, random_state = 42)
    
    return [Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest];

def split3(Xn, yn, m, mvalid, mtest): # noise + reduced dataset
    
    import random
    
    # dataset
    n = random.sample(range(1, Xn.shape[0]), m)
    Xn = Xn[n,:] # Xn should be defined
    yn = yn[n]   # yn should be defined
    
    # split
    X1, Xtest, y1, ytest = train_test_split(Xn,yn, test_size = mtest, random_state = 42)
    Xtrain, Xvalid, ytrain, yvalid = train_test_split(X1,y1, test_size = mvalid, random_state = 42)
    
    return [Xtrain, ytrain, Xvalid, yvalid, Xtest, ytest];

def CNN1D(n_timesteps,n_features):
    
    model = Sequential()
    
    # Conv layers
    model.add(Conv1D(filters=16, kernel_size=3, input_shape=(n_timesteps,n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Flatten())
    
    # Dense layer
    model.add(Dense(16, activation='relu'))
    
    # output layers
    model.add(Dense(1, activation='sigmoid'))
    
    return model

def train(Xtrain, ytrain, Xvalid, yvalid, model, alpha, nepoch, batchsize):
    
    # otherparameters
    m_train = Xtrain.shape[0]
    m_valid = Xvalid.shape[0]
    
    # reshape the array
    Xtrain = np.reshape(Xtrain, (m_train,n_timesteps,n_features)) # n_timesteps and n_feature = defined
    Xvalid = np.reshape(Xvalid, (m_valid,n_timesteps,n_features))
    
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4), metrics=['acc'])
    
    # train the model
    
    history = model.fit(Xtrain, ytrain, validation_data=(Xvalid, yvalid), epochs=nepoch, batch_size = batchsize, verbose=2, shuffle = True)
    
    return [model, history];

def lossplot(history, lb, ub, font, ep1, ep2):
    
    # Summarize history for loss...
    plt.figure(figsize=(14,8))
    plt.plot(history.history['loss'],'-o')
    plt.plot(history.history['val_loss'],'-s')
    plt.title('Loss curve for 1D-CNN',fontsize=font+2)
    plt.ylabel('Binary cross entropy loss',fontsize=font)
    plt.xticks(fontsize=font)
    plt.yticks(fontsize=font)
    plt.xlabel('Number of epochs',fontsize=font)
    plt.legend(['train', 'valid'], loc='upper right',fontsize=font)
    plt.axis([ep1,ep2,lb,ub])
    return plt.show()

def accplot(history, lb, ub, font, ep1, ep2):
    
    # Summarize history for acc...
    plt.figure(figsize=(14,8))
    plt.plot(history.history['acc'],'-o')
    plt.plot(history.history['val_acc'],'-s')
    plt.title('Accuracy curve for 1D-CNN',fontsize=font+2)
    plt.ylabel('Accuracy',fontsize=font)
    plt.xticks(fontsize=font)
    plt.yticks(fontsize=font)
    plt.xlabel('Number of epochs',fontsize=font)
    plt.legend(['train', 'valid'], loc='upper right',fontsize=font)
    plt.axis([ep1,ep2,lb,ub])
    return plt.show()

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
    
    m_test = Xtest.shape[0]    
    Xtest = np.reshape(Xtest,(m_test,n_timesteps,n_features))
    y_predicted = model.predict_classes(Xtest, verbose = 2)
    y_actual = ytest
    
    cm = confusion_matrix(y_actual, y_predicted)
    cm_labels = ['Undamaged','Damaged']
    
    return plot_confusion_matrix(cm, classes=cm_labels, title='Confusion matrix'), print(classification_report(y_actual, y_predicted, target_names = cm_labels))

#------------------------------------------imports----------------------------
%matplotlib inline
import random 
import tensorflow as tf

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Activation, MaxPooling1D, Dropout, Lambda 
from tensorflow.keras.layers import Dense, Conv1D, SimpleRNN, LSTM
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical, plot_model
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if (logs.get('val_loss')<1e-8) and (logs.get('loss')<1e-8):
      print("\nReached perfect accuracy so cancelling training!")
      self.model.stop_training = True

epoch_schedule = myCallback()

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 5))

df_UD_Ax = pd.read_csv('D:/SDE/1_SDEisotropicDL/CollectData/0_DataSet/Ax2500_UD_03Aug20.txt',header=None)
df_UD_Flex = pd.read_csv('D:/SDE/1_SDEisotropicDL/CollectData/0_DataSet/Flex2500_UD_03Aug20.txt',header=None)
df_D_Ax = pd.read_csv('D:/SDE/1_SDEisotropicDL/CollectData/0_DataSet/Ax2500_D_03Aug20.txt',header=None)
df_D_Flex = pd.read_csv('D:/SDE/1_SDEisotropicDL/CollectData/0_DataSet/Flex2500_D_03Aug20.txt',header=None)

nfft = 1024*8
T = 0.5e-6*nfft
q = 100e3
deltaT = T/nfft
t = np.arange(0,(nfft-1)/nfft,1/nfft)
time = t*T
print(time)

plt.figure(figsize=(10,4))
N = random.randint(1, 2500)
plt.plot(time, df_UD_Ax.iloc[N, 0 : df_UD_Ax.shape[1]-1])
plt.title('Undamaged Ax sample',fontsize=22)
print(N)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Time is seconds',fontsize=18)
plt.ylabel('Amplitude',fontsize=18)

plt.figure(figsize=(10,4))
plt.plot(time, df_UD_Flex.iloc[N, 0 : df_UD_Flex.shape[1]-1])
plt.title('Undamaged Flex sample',fontsize=22)
print(N)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Time is seconds',fontsize=18)
plt.ylabel('Amplitude',fontsize=18)

plt.figure(figsize=(10,4))
plt.plot(time, df_D_Ax.iloc[N, 0 : df_D_Ax.shape[1]-1])
plt.title('Damaged Ax sample',fontsize=22)
print(N)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Time is seconds',fontsize=18)
plt.ylabel('Amplitude',fontsize=18)

plt.figure(figsize=(10,4))
plt.plot(time, df_D_Flex.iloc[N, 0 : df_D_Flex.shape[1]-1])
plt.title('Damaged Flex sample',fontsize=22)
print(N)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Time is seconds',fontsize=18)
plt.ylabel('Amplitude',fontsize=18)

df_D = df_D_Ax + df_D_Flex
df_UD = df_UD_Ax + df_UD_Flex

plt.figure(figsize=(10,4))
plt.plot(time, df_UD.iloc[N, 0 : df_UD.shape[1]-1])
plt.title('UnDamaged sample',fontsize=22)
print(N)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Time is seconds',fontsize=18)
plt.ylabel('Amplitude',fontsize=18)

plt.figure(figsize=(10,4))
plt.plot(time, df_D.iloc[N, 0 : df_D.shape[1]-1])
plt.title('Damaged sample',fontsize=22)
print(N)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlabel('Time is seconds',fontsize=18)
plt.ylabel('Amplitude',fontsize=18)

index = np.where(time == 1e-03)
index = int(index[0])
print(index)

# Input/Features and labels extraction
DAx = df_D_Ax.iloc[:, 0 : index]
DFlex = df_D_Flex.iloc[:, 0 : index]
Dam = DAx + DFlex
# Dam = np.concatenate([DAx,DFlex], axis=1)
Dam = np.array(Dam)
Dam.shape

# Input/Features and labels extraction
UDAx = df_UD_Ax.iloc[:, 0 : index]
UDFlex = df_UD_Flex.iloc[:, 0 : index]
UDam = UDAx + UDFlex
#UDam = np.concatenate([UDAx,UDFlex], axis=1)
UDam = np.array(UDam)
UDam.shape

# Input/Features and labels extraction
X = np.concatenate([UDam,Dam], axis=0)
print(X.shape)

y_UD = np.zeros((len(UDam),1), dtype=int)
y_D = np.ones((len(Dam),1), dtype=int)
print(y_UD.shape)
print(y_D.shape)

# Input/Features and labels extraction
y = np.concatenate([y_UD,y_D], axis=0)
y = np.array(y)
print(y.shape)

n_timesteps = index
n_features = 1

# call split function
[Xtrain,ytrain,Xvalid,yvalid,Xtest,ytest] = split1(X, y, 0.2, 0.05)
print(Xtrain.shape)
print(Xtest.shape)

# call model architecture
model = CNN1D(n_timesteps,n_features)
model.summary()

# call train function
[model, history] = train(Xtrain, ytrain, Xvalid, yvalid, model, 1e-4, 250, 64)

# call lossplot function
lossplot(history, 0.5, 1e-7, 22, 0, 250)
lossplot(history, 1e-5, 1e-7, 22, 100, 250)
# call accplot function
accplot(history, 0.9, 1.0, 22, 0, 250)

# call pred function
pred(model,Xtest,ytest)

# weights
layer_weights = model.layers[0].get_weights()[0]
print(layer_weights.shape)
print(layer_weights)

# biases
layer_biases  = model.layers[0].get_weights()[1]
print(layer_biases.shape)
print(layer_biases)

#---Random gaussian noise parameter
beta1 = 0.01
beta2 = 0.02
beta3 = 0.03
mu = 0
sigma = 1
r = sigma*np.random.randn(n_timesteps,1) + mu   #random parameter with gaussian distribution
r = np.transpose(r)

#---Noisy signal
n1 = beta1*r*np.max(X)
n2 = beta2*r*np.max(X)
n3 = beta3*r*np.max(X)
Xn1 = X + n1
Xn2 = X + n2
Xn3 = X + n3

#---Signal to noise ratio
import math

rms_Xn = np.sqrt(np.mean(Xn1**2))
Power_Xn = rms_Xn**2

rms_n = np.sqrt(np.mean(n1**2))
Power_n = rms_n**2

SNR_dB = 10*math.log10(Power_Xn/Power_n)
print("SNR : ",SNR_dB)

#---Plot non-noisy signal
N = random.randint(1, 5000)
print(N)
if N<2500:
    print('Undamaged')
else:
    print('Damaged')
plt.figure(figsize=(10,4))
plt.plot(time[0 : index], Xn2[N, 0 : index])
plt.title('Noise')

## New dataset 

Xn = np.concatenate([X,Xn1,Xn2,Xn3], axis=0)
yn = np.concatenate([y,y,y,y], axis=0)
print(Xn.shape)
print(yn.shape)

# call split function
[Xtrain,ytrain,Xvalid,yvalid,Xtest,ytest] = split2(Xn,yn,0.2,0.05)
print(Xtrain.shape)
print(Xtest.shape)

# call model architecture
model = CNN1D(n_timesteps,n_features)
model.summary()

# call train function
[model, history] = train(Xtrain, ytrain, Xvalid, yvalid, model, 1e-4, 250, 64)

# call lossplot function
lossplot(history, 0.8, 1e-7, 22, 0, 250)
lossplot(history, 1e-6, 1e-8, 22, 220, 250)
# call accplot function
accplot(history, 0.9, 1.0, 22, 0, 250)

# call pred function
pred(model,Xtest,ytest)

# call split function
[Xtrain,ytrain,Xvalid,yvalid,Xtest,ytest] = split3(Xn,yn,10000,0.2,0.05)
print(Xtrain.shape)
print(Xtest.shape)

# call model architecture
model = CNN1D(n_timesteps,n_features)
model.summary()

# call train function
[model, history] = train(Xtrain, ytrain, Xvalid, yvalid, model, 1e-4, 250, 64)

# call lossplot function
lossplot(history)
# call accplot function
accplot(history)

# call pred function
pred(model,Xtest,ytest)

# call split function
[Xtrain,ytrain,Xvalid,yvalid,Xtest,ytest] = split3(Xn, yn, 5000, 0.2, 0.05)
print(Xtrain.shape)
print(Xtest.shape)

# call model architecture
model = CNN1D(n_timesteps,n_features)
model.summary()

# call train function
[model, history] = train(Xtrain, ytrain, Xvalid, yvalid, model, 1e-4, 250, 16)

# call lossplot function
lossplot(history)
# call accplot function
accplot(history)

# call pred function
pred(model,Xtest,ytest)

# call split function
[Xtrain,ytrain,Xvalid,yvalid,Xtest,ytest] = split3(Xn, yn, 2500, 0.2, 0.05)
print(Xtrain.shape)
print(Xtest.shape)

# call model architecture
model = CNN1D(n_timesteps,n_features)
model.summary()

# call train function
[model, history] = train(Xtrain, ytrain, Xvalid, yvalid, model, 1e-4, 250, 32)

# call lossplot function
lossplot(history)
# call accplot function
accplot(history)

# call pred function
pred(model,Xtest,ytest)

# call split function
[Xtrain,ytrain,Xvalid,yvalid,Xtest,ytest] = split3(Xn, yn, 1250, 0.2, 0.05)
print(Xtrain.shape)
print(Xtest.shape)

# call model architecture
model = CNN1D(n_timesteps,n_features)
model.summary()

# call train function
[model, history] = train(Xtrain, ytrain, Xvalid, yvalid, model, 1e-4, 250, 16)

# call lossplot function
lossplot(history)
# call accplot function
accplot(history)

# call pred function
pred(model,Xtest,ytest)

# call split function
[Xtrain,ytrain,Xvalid,yvalid,Xtest,ytest] = split3(Xn, yn, 625, 0.2, 0.05)
print(Xtrain.shape)
print(Xtest.shape)

# call model architecture
model = CNN1D(n_timesteps,n_features)
model.summary()

# call train function
[model, history] = train(Xtrain, ytrain, Xvalid, yvalid, model, 1e-4, 250, 8)

# call lossplot function
lossplot(history)
# call accplot function
accplot(history)

# call pred function
pred(model,Xtest,ytest)

# call split function
[Xtrain,ytrain,Xvalid,yvalid,Xtest,ytest] = split3(Xn, yn, 300,0.20,0.05)
print(Xtrain.shape)
print(Xtest.shape)

# call model architecture
model = CNN1D(n_timesteps,n_features)
model.summary()

# call train function
[model, history] = train(Xtrain, ytrain, Xvalid, yvalid, model, 1e-4, 250, 8)

# call lossplot function
lossplot(history)
# call accplot function
accplot(history)

# call pred function
pred(model,Xtest,ytest)

# call split function
[Xtrain,ytrain,Xvalid,yvalid,Xtest,ytest] = split3(Xn, yn, 150,0.20,0.05)
print(Xtrain.shape)
print(Xtest.shape)

# call model architecture
model = CNN1D(n_timesteps,n_features)
model.summary()

# call train function
[model, history] = train(Xtrain, ytrain, Xvalid, yvalid, model, 1e-4, 250, 4)

# call lossplot function
lossplot(history)
# call accplot function
accplot(history)

# call pred function
pred(model,Xtest,ytest)

# call split function
[Xtrain,ytrain,Xvalid,yvalid,Xtest,ytest] = split3(Xn, yn, 75,0.20,0.05)
print(Xtrain.shape)
print(Xtest.shape)

# call model architecture
model = CNN1D(n_timesteps,n_features)
model.summary()

# call train function
[model, history] = train(Xtrain, ytrain, Xvalid, yvalid, model, 1e-4, 250, 4)

TrainLoss = [1.63e-10,1.5e-9,1.1e-9,3e-7,8e-7,1.25e-4,1.2e-4,6e-4,0.0012];
ValLoss = [2.54e-10,1.4e-9,1e-9,4e-7,1.5e-6,7.7e-4,0.0106,0.0013,0.0149];
m = [20000,10000,5000,2500,1250,625,300,150,75]

plt.figure(figsize=(20,8))
plt.plot(m,TrainLoss,'-o')
plt.plot(m,ValLoss,'-s')
plt.title('Learning Curve of 1D-CNN for damage detection',fontsize=18)
plt.ylabel('Binary cross entropy loss',fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Number of training examples',fontsize=18)
plt.legend(['train', 'Validation'], loc='upper right',fontsize=18)
plt.axis([75,20000,1e-10,1e-2])
