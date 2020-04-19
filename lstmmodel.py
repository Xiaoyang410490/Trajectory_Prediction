import tensorflow as tf
from dataload import *
from visualization import *

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu,True)

def univariate_data(inputset, targetset,start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = 150 - target_size

    for i in range(start_index,end_index):
        input_seq = []
        target_seq = []
        for number in range(i-history_size,i):
            input_seq.append(inputset[number])
        for number in range(i,i+target_size):
            target_seq.append(targetset[number])

        data.append(input_seq)
        labels.append(target_seq)

    return np.array(data),np.array(labels)


def train_valid_get():

    target_data, target_index, num_index, velocity, winkel,target_vel,target_angle = target_get('allAmsterdamerRing.pickle')
    input_data = input_get(target_index=target_index, num_index=num_index, file_path='allAmsterdamerRing.pickle',
                            velocity=velocity, winkel=winkel)

    train_split = math.floor(len(target_data)*0.9)
    history_size = 30
    target_size = 1

    x_train = []
    y_train = []
    x_valid = []
    y_valid = []

    for i in range(train_split):
        input = input_data[i]
        target = target_data[i]
        x_train_uni, y_train_uni = univariate_data(input, target, 0, None, history_size, target_size)
        x_train.append(x_train_uni)
        y_train.append(y_train_uni)

    for i in range(train_split, len(target_data)):
        input = input_data[i]
        target = target_data[i]
        x_valid_uni, y_valid_uni = univariate_data(input, target, 0, None, history_size, target_size)
        x_valid.append(x_valid_uni)
        y_valid.append(y_valid_uni)

    return x_train,y_train,x_valid,y_valid

x_train,y_train,x_valid,y_valid = train_valid_get()

one_kurve = x_train[0]

def model():

    multi_step_model = tf.keras.models.Sequential()
    multi_step_model.add(tf.keras.layers.LSTM(16,
                                              return_sequences=True,
                                              input_shape=one_kurve.shape[-2:]))
    multi_step_model.add(tf.keras.layers.LSTM(8, activation='elu'))
    multi_step_model.add(tf.keras.layers.Dense(2))

    multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

    return multi_step_model

def model2():

    multi_step_model = tf.keras.models.Sequential()
    multi_step_model.add(tf.keras.layers.GRU(16,
                                              return_sequences=True,
                                              input_shape=one_kurve.shape[-2:]))
    multi_step_model.add(tf.keras.layers.GRU(8))
    multi_step_model.add(tf.keras.layers.Dense(2))

    multi_step_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='mae')

    return multi_step_model


model = model()

Buffle_size = 105
Batch_size = 35
evaluation_interval = 200
epochs = 10

for i in range(len(x_train)):

     input_data = x_train[i]
     target_data = y_train[i]

     train_univariate = tf.data.Dataset.from_tensor_slices((input_data, target_data))
     train_univariate = train_univariate.cache().shuffle(Buffle_size).batch(Batch_size).repeat()

     history = model.fit(train_univariate,epochs=epochs,steps_per_epoch=evaluation_interval)

     plot_train_history(history)
