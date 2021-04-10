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
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

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
                

model = Sequential()
model.add(Dense(12, input_dim=3, activation='relu'))
model.add(Dense(36, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#model.fit(X, y, epochs=150, batch_size=10)
# Well i didnt know how to proceed after this..  :(