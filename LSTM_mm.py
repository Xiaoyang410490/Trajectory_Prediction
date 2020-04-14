import tensorflow as tf
from dataload_model1 import *

x_train,y_train,x_valid,y_valid = train_valid_get()
x_train = np.swapaxes(x_train,1,0)
y_train = np.swapaxes(y_train,1,0)
x_valid = np.swapaxes(x_valid,1,0)
y_valid = np.swapaxes(y_valid,1,0)
Batch_size = 35

simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=x_train[0].shape[-2:]),
    tf.keras.layers.Dense(2)
])

simple_lstm_model.compile(optimizer='adam', loss='mae')

evaluation_interval = 200
epochs = 10

for i in range(len(x_train)):
     input_data = x_train[i]
     target_data = y_train[i]
     print(np.shape(input_data))
     print(np.shape(target_data))

     train_univariate = tf.data.Dataset.from_tensor_slices((input_data, target_data))
     train_univariate = train_univariate.cache().batch(Batch_size).repeat()

     simple_lstm_model.fit(train_univariate,epochs=epochs,steps_per_epoch=evaluation_interval)


