import numpy as np
from data import loadcsv

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from matplotlib import pyplot as plt

model = tf.keras.Sequential()
# model.add(layers.Dense(64, activation='tanh'))
# model.add(layers.Dense(64, activation='tanh'))
model.add(layers.SimpleRNN(10))
model.add(layers.Dense(1, activation='linear'))

model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='mean_squared_error',
              metrics=['mae'])

data, labels, mean = loadcsv('TRD_Exchange.csv', 5)

train_data = data[:-250, :].reshape([-1, 1, 5])
train_labels = labels[:-250].reshape([-1, 1, 1])

test_data = data[-250:, :].reshape([-1, 1, 5])
test_labels = labels[-250:].reshape([-1, 1, 1])

model.fit(train_data, train_labels, epochs=10, batch_size=32)

model.evaluate(test_data, test_labels, batch_size=250)
pred = model.predict(test_data).reshape([-1])
print(np.corrcoef(pred, test_labels.reshape([-1])))

plt.plot(pred+mean, label='predictions')
plt.plot(test_labels.reshape([-1])+mean, label='test data')
plt.xlabel('days')
plt.ylabel('RMB')
plt.legend()
plt.show()