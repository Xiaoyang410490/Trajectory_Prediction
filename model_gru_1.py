import tensorflow as tf

class gru_model(tf.keras.Model):

    def __init__(self,enc_units, batch_sz):

        super(gru_model, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(2, activation='elu')
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.batchnormalization = tf.keras.layers.BatchNormalization()

    def call(self, x, hidden, mode):

        output, state_h = self.gru(x, initial_state = hidden)
        output_1 = self.batchnormalization(output)
        results  = self.fc(output_1)
        if mode == 'train':
            results = self.dropout(results)
        return results ,state_h

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

    def initialize_cell_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))