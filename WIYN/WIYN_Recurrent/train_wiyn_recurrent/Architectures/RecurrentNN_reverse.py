import numpy as np
import tensorflow
import tensorflow as tf
from tensorflow import keras
from keras import layers
import h5py
import numpy.linalg as lg
from keras import regularizers
from sklearn.model_selection import train_test_split
#import format_data as formatter

## Path to total data used for WIYN
h5f = h5py.File('/home/u5/jzariski/TelescopeNet-main/WIYN/WIYN_Recurrent/train_wiyn_recurrent/formatting/TotalDataWIYN_recurrent_with_weather_reverse.hdf5', 'r')

filtered_data = h5f.get('dataset1')
print(filtered_data.shape)


split_num = int(np.round(filtered_data.shape[0]*0.9))
print(split_num)

features = filtered_data[:,:,0:31]
outputs = filtered_data[:,:,31:33]

h5f.close()

## Indices of features to train on. List of features/indices can be found in formatting script
#feats = [0,1,2,3,4,7,8,18,19]

feats = [0,1,2,3,4,7,8,18,19]
## Cuts around the year 2022, so we train on 2014-2021 and test on 2022 
## Changing cutsize from default of 10 requires this to be altered
X_train_og, X_test_og, Y_train, Y_test = features[0:split_num,:,:], features[split_num:,:,:], outputs[0:split_num,:,:], outputs[split_num:,:,:]

X_train, X_test = X_train_og[:, :, feats], X_test_og[:, :, feats]

def create_nn_dense(bsize) -> tensorflow.keras.models.Model:

    model = keras.Sequential()
    #model.add(layers.Normalization())
    model.add(layers.GRU(256, return_sequences=True, activation='swish', recurrent_dropout=0.2, input_shape=(None, X_train.shape[-1])))
    model.add(layers.GRU(128, return_sequences=True, activation='swish', recurrent_dropout=0.2))
    model.add(layers.Dense(2, activation='linear'))
    model.compile(loss='huber_loss', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

    return model

es = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=7)
reduce_lr_loss = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, epsilon=0.0001, mode='min')

bsize = 1
deep = create_nn_dense(bsize)

deep.fit(X_train, Y_train, verbose=1, epochs=120, batch_size=bsize,callbacks=[es, reduce_lr_loss], validation_split=0.1)

deep.save('NN_recurrent_acqs_WIYN_AST_dropout_reverse.h5')



## Used for testing purposes at the end of training 

preds = deep.predict(X_test)

sum0, sum1, sum2, sum3 = [], [], [], []

for i in range(Y_test.shape[0]):
    for j in range(Y_test.shape[1]):
        if abs(Y_test[i,j,0]) > 1e-8:
            perror = (preds[i,j,0]-Y_test[i,j,0])/Y_test[i,j,0]
            sum0.append(abs(perror)*100)
            perror_tcs = (X_test_og[i,j,5]-Y_test[i,j,0])/Y_test[i,j,0]
            sum2.append(abs(perror_tcs)*100)
        if abs(Y_test[i,j,1]) > 1e-8:
            perror = (preds[i,j,1]-Y_test[i,j,1])/Y_test[i,j,1]
            sum1.append(abs(perror)*100)
            perror_tcs = (X_test_og[i,j,6]-Y_test[i,j,1])/Y_test[i,j,1]
            sum3.append(abs(perror_tcs)*100)
            
ferror0 = np.median(sum0)
ferror1 = np.median(sum1)
ferror2 = np.median(sum2)
ferror3 = np.median(sum3)


print('My model', ferror0, ferror1)
print('TCS comparison', ferror2, ferror3)


  

