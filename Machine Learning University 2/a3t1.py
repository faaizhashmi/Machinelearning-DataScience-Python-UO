# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 10:19:27 2019

@author: Natur
"""
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import mnist
from skimage import io
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras import optimizers, metrics, regularizers, utils
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


batch_size =512
num_classes = 3
epochs = 10
model_name='thyroidInputs.xlsx'
model_name1='thyroidTargets.xlsx'


dataset = pd.read_excel(model_name,header=None)
dataset1=pd.read_excel(model_name1,header=None)
X = dataset.loc[:].values
y = dataset1.loc[:].values

df=pd.DataFrame(X)
X=df.T
df=pd.DataFrame(y)
y=df.T

# input image dimensions
#img_rows, img_cols = 28, 28

# the data, split between train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#initializing the ANN
classifier=Sequential()

#Adding input layer and first hidden layer
classifier.add(Dense(output_dim=20,init='uniform',activation='relu',input_dim=21))


#adding output layer
classifier.add(Dense(output_dim=3,init='uniform',activation='relu'))

#compiling our neural network        
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fitting our dataset to NN
history=classifier.fit(X_train, y_train, batch_size=7200, epochs=10)

#predicting test set results
y_pred=classifier.predict(X_test)
#y_pred=(y_pred>.01)

#confusion matrix
#from sklearn.metrics import confusion_matrix
#cm=confusion_matrix(y_test,y_pred)





# convert class vectors to b
#sgd = optimizers.SGD(lr=0.01, momentum=0.5)
opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)





# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


import sys
orig_stdout = sys.stdout
f = open(model_name+'_summary.txt', 'w')
sys.stdout = f
print(classifier.summary())
sys.stdout = orig_stdout
f.close()


score = classifier.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



