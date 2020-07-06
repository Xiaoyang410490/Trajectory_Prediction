import tensorflow as tf

class multi_lstm(tf.keras.Model):

    def __init__(self,enc_units, batch_sz):

        super(multi_lstm, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Dense(32, activation='elu')
        self.embedding_1 = tf.keras.layers.Dense(32, activation='elu')
        self.lstm = tf.keras.layers.LSTM(self.enc_units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
        self.lstm_1 = tf.keras.layers.LSTM(self.enc_units,
                                         return_sequences=True,
                                         return_state=True,
                                         recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(2, activation='elu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.batchnormalization = tf.keras.layers.BatchNormalization()
        self.batchnormalization_1 = tf.keras.layers.BatchNormalization()

    def call(self, x, hidden, cell, hidden_1,cell_1,mode,hidden_vector,part):

        if part=='encoder':
            inp_x = x
            input_embedding = self.embedding(inp_x)
            input_embedding = self.batchnormalization(input_embedding)
        else:
            inp_x = tf.concat([tf.expand_dims(hidden_vector, 1), x], axis=-1)
            input_embedding = self.embedding_1(inp_x)
            input_embedding = self.batchnormalization_1(input_embedding)

        if mode == 'train':
            input_x = self.dropout(input_embedding)
        else:
            input_x = input_embedding

        output, state_h, state_c = self.lstm(input_x, initial_state = [hidden,cell])
        output_1,state_h_1,state_c_1 = self.lstm_1(output,initial_state=[hidden_1,cell_1])

        output_1 = self.batchnormalization(output_1)

        results = self.fc(output_1)

        return results,state_h,state_c,state_h_1,state_c_1

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

    def initialize_cell_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))