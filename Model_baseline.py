import tensorflow as tf

class lstm_model(tf.keras.Model):

    def __init__(self,enc_units, batch_sz):

        super(lstm_model, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.lstm = tf.keras.layers.LSTM(self.enc_units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(2, activation='elu')

    def call(self, x, hidden, cell):

        output, state_h, state_c = self.lstm(x, initial_state = [hidden,cell])
        output = self.fc(output)
        return output,state_h,state_c

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

    def initialize_cell_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class multi_lstm(tf.keras.Model):

    def __init__(self,enc_units, batch_sz):

        super(multi_lstm, self).__init__()
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
        self.fc = tf.keras.layers.Dense(2, activation='elu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.batchnormalization = tf.keras.layers.BatchNormalization()

    def call(self, x, hidden, cell,hidden_1,cell_1,mode):

        output, state_h, state_c = self.lstm(x, initial_state = [hidden,cell])
        output_1,state_h_1,state_c_1 = self.lstm_1(output,initial_state=[hidden_1,cell_1])

        output_1 = self.batchnormalization(output_1)
        results = self.fc(output_1)
        if mode == 'train':
            results = self.dropout(results)

        return results,state_h,state_c,state_h_1,state_c_1

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

    def initialize_cell_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))



class multi_lstm_3(tf.keras.Model):

    def __init__(self,enc_units, batch_sz):

        super(multi_lstm_3, self).__init__()
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
        self.lstm_2 = tf.keras.layers.LSTM(self.enc_units,
                                           return_sequences=True,
                                           return_state=True,
                                           recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(2, activation='elu')
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, x, hidden, cell,hidden_1,cell_1,hidden_2,cell_2,mode):

        output, state_h, state_c = self.lstm(x, initial_state = [hidden, cell])
        output_1,state_h_1,state_c_1 = self.lstm_1(output,initial_state=[hidden_1, cell_1])
        output_2, state_h_2, state_c_2 = self.lstm_2(output, initial_state=[hidden_2, cell_2])
        results = self.fc(output_2)

        return results,state_h,state_c,state_h_1,state_c_1,state_h_2,state_c_2

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

    def initialize_cell_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))