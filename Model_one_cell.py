import tensorflow as tf

class Encoder_decoder(tf.keras.Model):
    def __init__(self ,units ,batch_sz, output_size):

        super(Encoder_decoder, self).__init__()
        self.batch_sz = batch_sz
        self.units = units
        self.lstm = tf.keras.layers.LSTM(self.units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
        self.lstm_1 = tf.keras.layers.LSTM(self.units,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(output_size, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.3)

    def call(self, x, hidden, cell, hidden_1, cell_1,mode ,mode_1):

        if mode == 'encoder':
            output, state_h, state_c = self.lstm(x, initial_state = [hidden, cell])
            results, state_h_1, state_c_1 = self.lstm_1(output,initial_state=[hidden_1,cell_1])
        else:
            output, state_h, state_c = self.lstm(x, initial_state=[hidden, cell])
            output_1, state_h_1, state_c_1 = self.lstm_1(output, initial_state=[hidden_1, cell_1])

            results = self.fc(output_1)

            if mode_1 == 'train':
                results = self.dropout(results, training=True)

        return results, state_h, state_c, state_h_1, state_c_1

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.units))

    def initialize_cell_state(self):
        return tf.zeros((self.batch_sz, self.units))

    def initialize_hidden_state_1(self):
        return tf.zeros((self.batch_sz, self.units))

    def initialize_cell_state_1(self):
        return tf.zeros((self.batch_sz, self.units))

