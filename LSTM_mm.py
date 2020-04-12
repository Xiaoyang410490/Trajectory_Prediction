import tensorflow as tf
from dataload_model1 import *

target_data,target_index,num_index,velocity,winkel = target_data('allAmsterdamerRing.pickle')
input_data = input_data(target_index=target_index,num_index=num_index,file_path='allAmsterdamerRing.pickle',velocity=velocity,winkel=winkel)

train_split = 15
history_size = 30
target_size = 15

x_train = []
y_train = []
x_valid = []
y_valid = []

for i in range(train_split):
    input = input_data[i]
    target = target_data[i]
    x_train_uni, y_train_uni = univariate_data(input,target, 0, None, history_size,target_size)
    x_train.append(x_train_uni)
    y_train.append(y_train_uni)

for i in range(train_split,8):
    input = input_data[i]
    target = target_data[i]
    x_valid_uni, y_valid_uni = univariate_data(input,target, 0, None, history_size,target_size)
    x_valid.append(x_valid_uni)
    y_valid.append(y_valid_uni)

Batch_size = 15

simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64,return_sequences = True,input_shape=x_train[0].shape[-2:]),
    tf.keras.layers.LSTM(16,return_sequences = True),
    tf.keras.layers.LSTM(8),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(2)
])

simple_lstm_model.compile(optimizer='adam', loss='mae')

evaluation_interval = 200
epochs = 10

for i in range(len(x_train)):
     train_univariate = tf.data.Dataset.from_tensor_slices((x_train[i], y_train[i]))
     train_univariate = train_univariate.cache().batch(Batch_size).repeat()

     simple_lstm_model.fit(train_univariate,epochs=epochs,steps_per_epoch=evaluation_interval)
