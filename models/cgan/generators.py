import tensorflow as tf


def create_lstm_generator_large(hparams):

    ls = hparams['latent_size']
    cs = hparams['condition_size']

    inp1 = tf.keras.layers.Input((ls,))
    prep1 = tf.keras.layers.Dense(cs)(inp1)
    prep1 = tf.keras.layers.Reshape((cs, 1))(prep1)

    inp2 = tf.keras.layers.Input((cs,))
    prep2 = tf.keras.layers.Dense(cs)(inp2)
    prep2 = tf.keras.layers.Reshape((cs, 1))(prep2)

    x = tf.keras.layers.Concatenate()([prep1, prep2])

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(ls*16, return_sequences=True))(x)
    x = tf.keras.layers.UpSampling1D()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(ls*8, return_sequences=True))(x)
    x = tf.keras.layers.UpSampling1D()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(ls*4, return_sequences=True))(x)
    x = tf.keras.layers.UpSampling1D()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(ls*2, return_sequences=True))(x)
    x = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(hparams['output_seq_len'])(x)
    model = tf.keras.models.Model([inp1, inp2], out, name='generator')
    return model


if __name__ == '__main__':

    hp = {'latent_size': 5, 'output_seq_len': 24, 'condition_size': 11}

    model = create_lstm_generator_large(hp)
    model.summary()
