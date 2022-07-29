import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split 


df = pd.read_csv("Dataset_ffnn.csv")

df_col= ['Direction']

for i in range(len(df_col)):
    df[df_col[i]] = LabelEncoder().fit_transform(df[df_col[i]])

X = df.iloc[:,3:11].values
Y = df.iloc[:,11:].values


sc_X = MinMaxScaler()
X[:,:]=sc_X.fit_transform(X[:,:])

sc_Y = MinMaxScaler()
Y[:,:]=sc_Y.fit_transform(Y[:,:])

from joblib import dump, load
dump(sc_X, 'sc_X.bin', compress=True);
dump(sc_Y, 'sc_Y.bin', compress=True);



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
 

 def model(X_train,y_train,X_test,y_test,X):
    
    model = tf.keras.Sequential([keras.layers.Dense(units=50, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.0001),input_shape=(8,))])
    model.add(keras.layers.Dense(units=50, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001),input_shape=(50,)))
    model.add(keras.layers.Dense(units=50, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001),input_shape=(50,)))
    model.add(keras.layers.Dense(units=5,kernel_regularizer=tf.keras.regularizers.l2(0.01) ,input_shape=(50,)))
    
    checkpoint_filepath = 'C:/Users/User/Desktop/MLmodel/tmp/checkpoint'
    monitor = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,monitor='val_acc', min_delta=0, patience=10, 
        verbose=1, mode='max', save_weights_only=True,save_best_only=True)
    
    optimizer = tf.keras.optimizers.SGD(0.001)
    model.compile(optimizer= optimizer, loss='mean_absolute_error', metrics=['accuracy',tf.keras.metrics.RootMeanSquaredError()])
    hist = model.fit(X_train,y_train,  epochs=30000,callbacks=[monitor], verbose=2,validation_data=(X_test,y_test))
    model.summary(line_length=None, positions=None, print_fn=None)
    model.save("my_model")
    score = model.evaluate(X_test, Y_test, verbose = 1) 

    print('Test loss:', score[0]) 
    print('Test accuracy:', score[1])
    return model.predict(X), hist
  
  predict, hist = model(X_train,Y_train,X_test,Y_test,X)
  
train_loss = hist.history['loss']
train_acc  = hist.history['acc']
validation_loss = hist.history['val_loss']
validation_acc  = hist.history['val_acc']


