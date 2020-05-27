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
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
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

    def __init__(self, output_size, dec_units, batch_sz):
        super(Decoder_Attention, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(output_size,activation='relu')
        self.batchnormalization = tf.keras.layers.BatchNormalization()
        # used for attention
        self.attention = BahdanauAttention(self.dec_units)
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, x, hidden, enc_output, mode):

        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)
        # x shape  == (batch_size, 1, 6)
        # x shape after concatenation == (batch_size, 1, 8 + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        output = self.batchnormalization(output)

        results = self.fc(output)

        if mode == 'train':
            results = self.dropout(results,training = True)

        return results,state


class Decoder(tf.keras.Model):

    def __init__(self, output_size, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.embedding = tf.keras.layers.Dense(dec_units,activation='relu')
        self.fc = tf.keras.layers.Dense(output_size,activation='relu')
        self.batchnormalization = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.2)

    def call(self, x, hidden, enc_output, mode):

        # hidden== (batch_size, hidden_size)
        hidden_vector = hidden
        # x shape  == (batch_size, 1, 8)
        # x shape after embedding == (batch_size,1,hidden_size)
        # x shape after concatenation == (batch_size, 1, 2 * hidden_size)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(hidden_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        output = self.batchnormalization(output)

        results = self.fc(output)

        if mode == 'train':
            results = self.dropout(results,training = True)

        return results,state


