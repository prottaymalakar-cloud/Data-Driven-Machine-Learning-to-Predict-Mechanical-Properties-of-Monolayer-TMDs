#importing necessary modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import r2_score
import datetime
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from joblib import dump, load
from sklearn.model_selection import train_test_split

#Dataset input
df = pd.read_csv("Data.csv")
df_col= ['Direction']
for i in range(len(df_col)):
    df[df_col[i]] = LabelEncoder().fit_transform(df[df_col[i]])
    
X = df.iloc[:,0:6].values
Y = df.iloc[:,-1:].values

#feature Scaling and Reshaping
sc_X = MinMaxScaler()
X[:,:]=sc_X.fit_transform(X[:,:])
sc_Y = MinMaxScaler()
Y[:,:]=sc_Y.fit_transform(Y[:,:])
dump(sc_X, 'sc_X.bin', compress=True);
dump(sc_Y, 'sc_Y.bin', compress=True);
X1 = np.reshape(X,(288,301,6))
Y1 = np.reshape(Y,(288,301,1))


#Dividing dateset into training, test and validation sets
X2, X_test, Y2, y_test = train_test_split(X1, Y1, test_size=0.05, random_state=42)
X_train, X_dev, y_train, y_dev = train_test_split(X2, Y2, test_size=0.1, random_state=42)

#creating_model
def buildmodel(X_train,y_train,X_dev,y_dev,X_test,y_test):
    model = tf.keras.Sequential()
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=6,activation='tanh')))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=30,activation='tanh')))
    #model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=6,activation='tanh')))
    model.add(keras.layers.LSTM(units=30, activation='tanh',return_sequences = True,input_shape = (301,6)))
    model.add(keras.layers.LSTM(units=30, activation='tanh',return_sequences = True,input_shape = (301,6)))
    #model.add(keras.layers.Dense(units=50, activation='relu'))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=30, activation='tanh')))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=6, activation='tanh')))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=1, activation='linear')))
    
    checkpoint_filepath = 'tmp/checkpoint'
    monitor = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,monitor='loss', min_delta=0, patience=0, 
        verbose=1, mode='min', save_weights_only=True,save_best_only=True)
    #log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #monitor = keras.callbacks.TensorBoard('logs/', histogram_freq=1)
    model.compile(optimizer= 'adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    hist = model.fit(X_train,y_train, epochs=1000,batch_size = 64,callbacks=[monitor], verbose=2,validation_data=(X_dev,y_dev))
    print(model.summary())
    model.save("my_lstm")
    keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    val_mse,val_mae = model.evaluate(X_test,y_test)
    print(val_mse)
    print(val_mae)
    #model.get_layer(name=None, index=None)
    return model.predict(X_test),model.predict(X_dev),model.predict(X_train), hist
  
#training the model
predict_test,predict_dev,predict_train,hist = buildmodel(X_train,y_train,X_dev,y_dev,X_test,y_test)

#results: loss
train_loss = hist.history['loss']
#train_acc  = hist.history['acc']
validation_loss = hist.history['val_loss']
#validation_acc  = hist.history['val_acc']
epochs = range(np.shape(train_loss)[0])

plt.plot(epochs,train_loss,epochs,validation_loss,)

df1 = pd.DataFrame(train_loss) # A is a numpy 2d array
df1.to_csv("train_loss.csv", header=False,index=False)
df2 = pd.DataFrame(validation_loss) # A is a numpy 2d array
df2.to_csv("Validation_loss.csv", header=False,index=False)
