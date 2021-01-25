import tensorflow as tf


def create_lstm_generator_large(hparams):
    ls = hparams['latent_size']
    inp = tf.keras.layers.Input((ls, hparams['condition_size']))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(ls*16, return_sequences=True))(inp)
    x = tf.keras.layers.UpSampling1D()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(ls*8, return_sequences=True))(x)
    x = tf.keras.layers.UpSampling1D()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(ls*4, return_sequences=True))(x)
    x = tf.keras.layers.UpSampling1D()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(ls*2, return_sequences=True))(x)
    x = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(hparams['output_seq_len'])(x)
    model = tf.keras.models.Model(inp, out, name='generator')
    return model


if __name__ == '__main__':

    hp = {'latent_size': 5, 'output_seq_len': 24, 'condition_size': 11}

    model = create_lstm_generator_large(hp)
    model.summary()
