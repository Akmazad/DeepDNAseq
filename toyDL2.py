# example is based on "DL with Python book #CH5 - by Francois Chollet"
# Example train data is the "toyDat_Bundle_v1.0" (created by A.K.M. Azad)


def MCC(y_true, y_pred):
    '''Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    '''
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())
	



import numpy as np
#import h5py
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score	
from keras import backend as K
from keras import optimizers
import tensorflow as tf


''' Create the model : a network with 3 convolutional layers and a dense layer '''
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Input, Convolution2D, MaxPooling2D, Flatten
from keras.models import Model 
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils import class_weight

num_classes = 2
batch_size = 16
num_epochs = 5

model = Sequential()
model.add(Conv2D(32, kernel_size=(4,4),activation='relu',input_shape=(1000,4,1),padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((1, 2),padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (4, 4), activation='relu',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(1, 2),padding='same'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (4, 4), activation='relu',padding='same'))
model.add(LeakyReLU(alpha=0.1))                  
model.add(MaxPooling2D(pool_size=(1, 2),padding='same'))
model.add(Dropout(0.2))


# Adding a classifier on top of the convnet
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(LeakyReLU(alpha=0.1))                  
model.add(Dense(num_classes, activation='sigmoid'))

# Compile the Convnet
#model.compile(loss='binary_crossentropy',
#              optimizer='adam',
#              metrics=['accuracy'])
			  
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy',MCC])
			  
# Training the convnet on toy DNA sequences
from keras.utils import to_categorical
import toyDL_utils

# Load the training and test data 
(train_data, train_labels), (test_data, test_labels) = toyDL_utils.load_CSV_data()
#train_data = (np.arange(train_data.max()) == train_data[...,None]-1).astype(int)
train_data = (np.arange(train_data.max()) == train_data[...,None]-1).astype('float32')
train_data =  train_data.reshape(2047,1000,4,1)
test_data = (np.arange(test_data.max()) == test_data[...,None]-1).astype('float32')
test_data =  test_data.reshape(500,1000,4,1)
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
history = model.fit(train_data, train_labels, epochs=num_epochs, batch_size=batch_size)


# reshape: temp = train_data.reshape(2047,4,1000)
#train_data = train_data.astype('float32')
#test_data = test_data.astype('float32')
#train_labels = to_categorical(train_labels)
#test_labels = to_categorical(test_labels)
#model.compile(optimizer='rmsprop',
#loss='categorical_crossentropy',
#metrics=['accuracy'])

history = model.fit(train_data, train_labels, epochs=num_epochs, batch_size=batch_size)

# Let’s evaluate the model on the test data:
score = model.evaluate(test_data, test_labels, verbose=0)
score 

# Training history
#np.save('score_CNN_score_', score)
#np.save('toy_MCC_train_', np.asarray(history.history['MCC']))
#np.save('toy_MCC_val_', np.asarray(history.history['val_MCC']))

# testing
pred_test_labels = model.predict(test_data)
#roc_auc_scoreroc_auc(test_labels, pred_test_labels)
#average_precision_score(test_labels, pred_test_labels)
