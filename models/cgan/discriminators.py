import tensorflow as tf


def create_lstm_discriminator_large(hparams):
    ls = hparams['latent_size']
    os = hparams['output_seq_len']
    cs = hparams['condition_size']

    inp1 = tf.keras.layers.Input((os,))
    prep1 = tf.keras.layers.Reshape((os, 1))(inp1)

    inp2 = tf.keras.layers.Input((cs,))
    prep2 = tf.keras.layers.Dense(os)(inp2)
    prep2 = tf.keras.layers.Reshape((os, 1))(prep2)

    x = tf.keras.layers.Concatenate()([prep1, prep2])
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(ls*4, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(ls*8, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(ls*16, return_sequences=True))(x)
    x = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model([inp1, inp2], out, name='discriminator')

    return model


if __name__ == '__main__':

    hp = {'latent_size': 5, 'output_seq_len': 24, 'condition_size': 11}

    model = create_lstm_discriminator_large(hp)
    model.summary()
