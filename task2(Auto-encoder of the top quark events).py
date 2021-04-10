# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 13:54:30 2021

@author: sj
"""

import pandas as pd 
from keras import layers
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot

file=['Boosted_Jets_Sample-0.snappy.parquet','Boosted_Jets_Sample-1.snappy.parquet','Boosted_Jets_Sample-2.snappy.parquet','Boosted_Jets_Sample-3.snappy.parquet','Boosted_Jets_Sample-4.snappy.parquet']
X=[[],[],[]]
for i in file:
    df=pd.read_parquet(i, engine='pyarrow')
    for i in df:
        for j in df[i]:
            for k in range(0,len(j)):
                if k==0:
                    X[0].append(k[j])
                if k==1:
                    X[1].append(k[j])
                if k==2:
                    X[2].append(k[j])
                
    
n_inputs = X.shape[1]
# scale data
t = MinMaxScaler()
t.fit(X)
X_train = t.transform(X)
visible = Input(shape=(n_inputs,))
e = Dense(n_inputs*2)(visible)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
e = Dense(n_inputs)(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)
n_bottleneck = n_inputs
bottleneck = Dense(n_bottleneck)(e)
d = Dense(n_inputs)(bottleneck)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
d = Dense(n_inputs*2)(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)
# output layer
output = Dense(n_inputs, activation='linear')(d)
model = Model(inputs=visible, outputs=output)
model.compile(optimizer='adam', loss='mse')
plot_model(model, 'autoencoder.png', show_shapes=True)
history = model.fit(X, X, epochs=200, batch_size=16)
pyplot.plot(history.history['loss'], label='train')

encoder = Model(inputs=visible, outputs=bottleneck)
plot_model(encoder, 'encoder_no_compress.png', show_shapes=True)
# save the encoder to file for future use
encoder.save('encoder.h5')

