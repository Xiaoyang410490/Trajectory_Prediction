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

def train_valid_get2():

    target_data, target_index, num_index, velocity, winkel,target_vel,target_angle = target_get('allAmsterdamerRing.pickle')
    input_data = input_get(target_index=target_index, num_index=num_index, file_path='allAmsterdamerRing.pickle',
                            velocity=velocity, winkel=winkel)

    train_split = math.floor(len(target_data)*0.9)
    history_size = 30
    target_size = 15

    x_train = []
    y_train = []
    x_valid = []
    y_valid = []

    x_train1 = []
    y_train1 = []
    x_valid1 = []
    y_valid1= []

    for i in range(train_split):
        input = input_data[i]
        target_velocity = target_vel[i]
        target_winkel = target_angle[i]
        x_train_uni1, y_train_uni1 = univariate_data(input, target_velocity, 0, None, history_size, target_size)
        x_train_uni2, y_train_uni2 = univariate_data(input, target_winkel, 0, None, history_size, target_size)
        x_train.append(x_train_uni1)
        y_train.append(y_train_uni1)
        x_train1.append(x_train_uni2)
        y_train1.append(y_train_uni2)

    for i in range(train_split, len(target_data)):
        input = input_data[i]
        target_velocity = target_vel[i]
        target_winkel = target_angle[i]
        x_valid_uni1, y_valid_uni1 = univariate_data(input, target_velocity, 0, None, history_size, target_size)
        x_valid_uni2, y_valid_uni2 = univariate_data(input, target_winkel, 0, None, history_size, target_size)
        x_valid.append(x_valid_uni1)
        y_valid.append(y_valid_uni1)
        x_valid1.append(x_valid_uni2)
        y_valid1.append(y_valid_uni2)

    return x_train,y_train,x_valid,y_valid,x_train1,y_train1,x_valid1,y_valid1


x_train,y_train,x_valid,y_valid,x_train1,y_train1,x_valid1,y_valid1 = train_valid_get2()

one_kurve = x_train[0]

def model3():

    multi_step_model = tf.keras.models.Sequential()
    multi_step_model.add(tf.keras.layers.LSTM(64,
                                              return_sequences=True,
                                              input_shape=one_kurve.shape[-2:]))
    multi_step_model.add(tf.keras.layers.LSTM(32, activation='elu'))
    multi_step_model.add(tf.keras.layers.Dense(15))

    multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

    return multi_step_model

model = model3()
Buffle_size = 105
Batch_size = 35
evaluation_interval = 200
epochs = 10

print(np.shape(x_train))

for i in range(2):

    input_train = x_train[i]
    target_train = y_train[i]

    input_valid = x_valid[0]
    target_valid = y_valid[0]

    train_univariate = tf.data.Dataset.from_tensor_slices((input_train, target_train))
    train_univariate = train_univariate.cache().batch(Batch_size).repeat()

    valid_univariate = tf.data.Dataset.from_tensor_slices((input_valid, target_valid))
    valid_univariate = valid_univariate.cache().batch(Batch_size)

    history = model.fit(train_univariate, epochs=epochs, steps_per_epoch=evaluation_interval,validation_data=valid_univariate)

    print('\nhistory dict:', history.history)

prediction=model.predict(x_valid[0])
print(prediction)


