import numpy as np
from data import loadcsv

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = tf.keras.Sequential()
model.add(layers.Dense(64, activation='tanh'))
model.add(layers.Dense(64, activation='tanh'))
model.add(layers.Dense(1, activation='linear'))

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='mean_squared_error',
              metrics=['mae'])

data, labels, mean = loadcsv('TRD_Exchange.csv', 5)

train_data = data[:-250, :]
train_labels = labels[:-250]

test_data = data[-250:, :]
test_labels = labels[-250:]

model.fit(train_data, train_labels, epochs=100, batch_size=32)

model.evaluate(test_data, test_labels, batch_size=250)
pred = model.predict(test_data).reshape([-1])
print(np.correlate(pred, test_labels))