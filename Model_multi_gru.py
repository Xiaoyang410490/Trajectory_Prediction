import tensorflow as tf

class Encoder(tf.keras.Model):

    def __init__(self,enc_units, batch_sz):

        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):

        output, state = self.gru(x, initial_state = hidden)

        return output,state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units,activation='relu')
        self.W2 = tf.keras.layers.Dense(units,activation='relu')
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder_Attention(tf.keras.Model):

    def __init__(self, output_size, dec_units,batch_sz):
        super(Decoder_Attention, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.gru_1 = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(output_size,activation='relu')
        self.batchnormalization = tf.keras.layers.BatchNormalization()
        # used for attention
        self.attention = BahdanauAttention(16)
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, x, hidden, enc_output, mode):

        # passing the concatenated vector to the GRU
        context_vector, attention_weights = self.attention(hidden, enc_output)
        concat_vector = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state = self.gru(concat_vector)

        output_1,state_1 = self.gru_1(output)

        output_2 = self.batchnormalization(output_1)

        results = self.fc(output_2)

        if mode == 'train':
            results = self.dropout(results,training = True)

        return results,state


class Decoder(tf.keras.Model):

    def __init__(self, output_size, dec_units,batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Dense(dec_units, activation='relu')
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.gru_1 = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(output_size,activation='relu')
        self.batchnormalization = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, x, hidden, hidden_1, enc_output, mode):

        hidden_vector = hidden
        hidden_vector_1 = hidden_1
        x = self.embedding(x)

        concat_vector = tf.concat([tf.expand_dims(hidden_vector, 1), x], axis=-1)
        output, state = self.gru(concat_vector)

        concat_vector_1 = tf.concat([tf.expand_dims(hidden_vector_1, 1), output], axis=-1)
        output_1,state_1 = self.gru_1(concat_vector_1)

        output_2 = self.batchnormalization(output_1)

        results = self.fc(output_2)

        if mode == 'train':
            results = self.dropout(results,training = True)

        return results,state,state_1

    def initialize_hidden_state_1(self):
        return tf.zeros((self.batch_sz, self.dec_units))