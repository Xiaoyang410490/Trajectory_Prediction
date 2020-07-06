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

    def call(self, x, hidden,cell):

        output, state,cell = self.lstm(x, initial_state = [hidden,cell])

        return output,state,cell

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
        self.embedding = tf.keras.layers.Dense(32,activation='elu')
        self.fc = tf.keras.layers.Dense(output_size,activation='elu')
        self.batchnormalization = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, x, enc_hidden, hidden,cell, enc_output, mode):

        # hidden== (batch_size, hidden_size)
        hidden_vector = enc_hidden
        # x shape  == (batch_size, 1, 8)
        # x shape after embedding == (batch_size,1,hidden_size)
        # x shape after concatenation == (batch_size, 1, 2 * hidden_size)

        x = tf.concat([tf.expand_dims(hidden_vector, 1), x], axis=-1)

        input_embedding = self.embedding(x)

        if mode == 'train':
            input_x = self.dropout(input_embedding)
        else:
            input_x = input_embedding

        # passing the concatenated vector to the GRU
        output, state, cell = self.lstm(input_x,initial_state=[hidden,cell])

        output = self.batchnormalization(output)

        results = self.fc(output)

        return results,state,cell