import tensorflow as tf

class Encoder_lstm(tf.keras.Model):

    def __init__(self,enc_units, batch_sz):

        super(Encoder_lstm, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.lstm = tf.keras.layers.LSTM(self.enc_units,
                                         return_sequences=True,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform')

    def call(self, x, hidden, cell):

        output, state_h, state_c = self.lstm(x, initial_state = [hidden, cell])
        return output, state_h, state_c

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

    def initialize_cell_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class Decoder_lstm(tf.keras.Model):

    def __init__(self, output_size, dec_units, dec_units_1, batch_sz):
        super(Decoder_lstm, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.dec_units_1 = dec_units_1
        self.lstm = tf.keras.layers.LSTM(self.dec_units,
                                         return_sequences=True,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform')
        self.lstm_1 = tf.keras.layers.LSTM(self.dec_units_1,
                                         return_sequences=True,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(output_size,activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.3)

    def call(self, x, hidden, cell, hidden_1,cell_1, mode):


        output, state, cell_state = self.lstm(x, initial_state=[hidden,cell])

        output_1,state_1,cell__state_1 = self.lstm_1(output,initial_state=[hidden_1,cell_1])

        results = self.fc(output_1)

        if mode == 'train':
            results = self.dropout(results,training = True)

        return results,state,cell_state,state_1,cell__state_1

    def initialize_hidden_state_1(self):
        return tf.zeros((self.batch_sz, self.dec_units_1))

    def initialize_cell_state_1(self):
        return tf.zeros((self.batch_sz, self.dec_units_1))