# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 14:12:00 2021

@author: sj
"""

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
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from normalizing_flows.flows import Flow, affine
from normalizing_flows.models.losses import kl_divergence_normal
from tqdm import tqdm
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras import Model
from normalizing_flows.layers import FlowLayer


file=['Boosted_Jets_Sample-0.snappy.parquet','Boosted_Jets_Sample-1.snappy.parquet','Boosted_Jets_Sample-2.snappy.parquet','Boosted_Jets_Sample-3.snappy.parquet','Boosted_Jets_Sample-4.snappy.parquet']
x_target=[[],[],[]]
for i in file:
    df=pd.read_parquet(i, engine='pyarrow')
    for i in df:
        for j in df[i]:
            for k in range(0,len(j)):
                if k==0:
                    x_target[0].append(k[j])
                if k==1:
                    x_target[1].append(k[j])
                if k==2:
                    x_target[2].append(k[j])
                
    
n_epochs = 50
batch_size = 100
N=len(x_target)
input_0 = Input(shape=(2,))
h = Dense(32, activation='relu')(input_0)
h = Dense(64, activation='relu')(h)
h = Dense(16, activation='relu')(h)
mu = Dense(2)(h)
log_var = Dense(2)(h)
flow_layer = FlowLayer()
zs,aa,aa1 = flow_layer([mu, log_var])
model = Model(inputs=input_0, outputs=zs[-1])
model.compile(loss='mse', optimizer='adam', metrics=['mse'])
betas = np.linspace(1.0E-3, 1.0, n_epochs*N//batch_size)

def update_beta(batch, logs):
    flow_layer.set_beta(betas[batch])
callback = LambdaCallback(on_batch_end=update_beta)
model.fit(x_target, x_target, batch_size=batch_size, epochs=n_epochs, callbacks=[callback])

x_pred = model.predict(x_target)

plt.figure(figsize=(6,4))
plt.scatter(x_target,x_target)
plt.scatter(x_pred, x_pred)
plt.legend(['Target', 'Estimated'])
plt.title('side-by side comparison of the original and reconstructed events')
plt.show()

