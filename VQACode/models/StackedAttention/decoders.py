import tensorflow as tf 


class GRUDecoder(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, units, vocab_size):
        super(GRUDecoder, self).__init__() 
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units, 
                                        return_sequences=True,
                                        return_state=True, 
                                        recurrent_initializer='glorot_uniform') 
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size) 

    def call(self, x, context, hidden): 
        x = self.embedding(x)

        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        output, state = self.gru(x)

        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2]))

        x = self.fc2(x)

        return x, state 
    
    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units)) 