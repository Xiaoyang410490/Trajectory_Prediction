import tensorflow as tf

class Encoder(tf.keras.Model):

    def __init__(self,enc_units, batch_sz):

        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.lstm = tf.keras.layers.LSTM(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.lstm_1 = tf.keras.layers.LSTM(self.enc_units,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer='glorot_uniform')
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, x, hidden, cell, hidden_1, cell_1,mode):

        if mode == 'train':
            input_x = self.dropout(x)
        else:
            input_x = x

        output, state_h, state_c = self.lstm(input_x, initial_state=[hidden, cell])
        output_1, state_h_1, state_c_1 = self.lstm_1(output, initial_state=[hidden_1, cell_1])

        return output_1,state_h,state_c,state_h_1,state_c_1

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

    def initialize_cell_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class Decoder(tf.keras.Model):

    def __init__(self, output_size, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.lstm = tf.keras.layers.LSTM(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.lstm_1 = tf.keras.layers.LSTM(self.dec_units,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer='glorot_uniform')
        self.embedding = tf.keras.layers.Dense(32,activation='elu')
        self.fc = tf.keras.layers.Dense(output_size,activation='elu')
        self.batchnormalization = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, x, enc_hidden, hidden, cell, hidden_1, cell_1, enc_output, mode):


        hidden_vector = enc_hidden

        x = tf.concat([tf.expand_dims(hidden_vector, 1), x], axis=-1)

        input_embedding = self.embedding(x)

        if mode == 'train':
            input_x = self.dropout(input_embedding)
        else:
            input_x = input_embedding

        # passing the concatenated vector to the GRU
        output, state, cell = self.lstm(input_x,initial_state=[hidden,cell])
        output_1, state_h_1, state_c_1 = self.lstm_1(output, initial_state=[hidden_1, cell_1])

        output_2 = self.batchnormalization(output_1)

        results = self.fc(output_2)

        return results,state,cell,state_h_1,state_c_1