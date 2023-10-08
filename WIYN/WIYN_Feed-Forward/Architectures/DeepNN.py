import numpy as np
import tensorflow
import tensorflow as tf
from tensorflow import keras
from keras import layers
import h5py
from keras import regularizers
from sklearn.model_selection import train_test_split

## Procedure for creating and training a pointing correction network
## Input files can be created from scripts in the formatting section

## Specify innput file with training and output data arranged as dataframe
h5f = h5py.File('/home/u5/jzariski/TelescopeNet-main/WIYN/WIYN_Feed-Forward/formatting/TotalDataWIYN_with_weather.hdf5', 'r')

filtered_data = h5f.get('dataset1')

## Chan change these indices to specify training features and output targets
features = filtered_data[:,0:31]
outputs = filtered_data[:,31:33]

## Can change the fractional split for training/test data here
split_num = int(np.round(filtered_data.shape[0]*0.9))


## Splitting procedure
X_train_og, X_test_og, Y_train, Y_test = features[0:split_num,:], features[split_num:,:], outputs[0:split_num,:], outputs[split_num:,:]

## Feature reduction can also take place here
feats = [0,1,2,3,4,7,8,18,19,20]

X_train, X_test = X_train_og[:, feats], X_test_og[:, feats]


## Creates a dense neural network. Hyperparameters can be tuned here, notably neuron count, act. functions and optimizer
def create_nn_dense(input_shape: tuple, n_outputs: int,) -> tensorflow.keras.models.Model:

    model = keras.Sequential()

    model.add(layers.Dense(units=1024, input_dim=input_shape[1], activation='tanh'))
    model.add(layers.Dense(512, activation='tanh'))
    model.add(layers.Dense(n_outputs, activation='linear'))

    model.compile(loss = 'huber_loss', optimizer = 'adam', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model
    
## Adding in early stopping and learning rate reduction on plateaus of training
es = tensorflow.keras.callbacks.EarlyStopping(monitor='loss', mode='auto', verbose=1, patience=7)
reduce_lr_loss = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, verbose=1, epsilon=0.0001, mode='min')

## Process for training the network. Can further tune some hyperparameters
deep = create_nn_dense(X_train.shape, 2)
deep.fit(X_train, Y_train, verbose=1, epochs=120, batch_size=64, callbacks=[es, reduce_lr_loss], validation_split=0.1)

deep.save('NN_deep_acqs_WIYN.h5')


## Brief testing procedure
## Medians give a relative guess as to how a model will compare to another
## Can remove if desired
preds = deep.predict(X_test)

sum0, sum1, sum2, sum3 = [], [], [], []

for i in range(len(Y_test[:,0])):
    if abs(Y_test[i,0]) > 1e-8:
        perror = (preds[i,0]-Y_test[i,0])/Y_test[i,0]
        sum0.append(abs(perror)*100)
        perror_tcs = (X_test_og[i,5]-Y_test[i,0])/Y_test[i,0]
        sum2.append(abs(perror_tcs)*100)
    if abs(Y_test[i,1]) > 1e-8:
        perror = (preds[i,1]-Y_test[i,1])/Y_test[i,1]
        sum1.append(abs(perror)*100)
        perror_tcs = (X_test_og[i,6]-Y_test[i,1])/Y_test[i,1]
        sum3.append(abs(perror_tcs)*100)
        
ferror0 = np.median(sum0)
ferror1 = np.median(sum1)
ferror2 = np.median(sum2)
ferror3 = np.median(sum3)

print('My model', ferror0, ferror1)
print('TCS comparison', ferror2, ferror3)


    





