import tensorflow as tf
import tensorflow_addons as tfa


def create_conv_discriminator_simple(hparams):
    inp = tf.keras.layers.Input((hparams['output_seq_len'],))
    x = tf.keras.layers.Reshape((hparams['output_seq_len'], 1))(inp)
    x = tf.keras.layers.Conv1D(32, 3, padding='same', strides=2, activation='relu')(x)
    x = tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(32, 3, padding='same', strides=2, activation='relu')(x)
    x = tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(1)(x)

    model = tf.keras.models.Model(inp, out, name='discriminator')
    return model


def create_lstm_discriminator_simple(hparams):
    ls = hparams['latent_size']
    inp = tf.keras.layers.Input((hparams['output_seq_len'],))
    x = tf.keras.layers.Reshape((hparams['output_seq_len'], 1))(inp)
    x = tf.keras.layers.LSTM(ls*2, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(ls*4, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(ls*8, return_sequences=True)(x)
    x = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(inp, out, name='discriminator')
    return model


def create_lstm_discriminator_large(hparams):
    ls = hparams['latent_size']
    inp = tf.keras.layers.Input((hparams['output_seq_len'],))
    x = tf.keras.layers.Reshape((hparams['output_seq_len'], 1))(inp)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(ls*4, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(ls*8, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(ls*16, return_sequences=True))(x)
    x = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.models.Model(inp, out, name='discriminator')
    return model


def create_conv_critic_complex(hparams):

    def block(x):
        c = tf.keras.layers.Conv1D(32, 3, padding='same')
        x = tfa.layers.SpectralNormalization(c, power_iterations=10)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.MaxPool1D(pool_size=2)(x)
        return x

    inp = tf.keras.layers.Input((hparams['output_seq_len'],))
    x = tf.keras.layers.Reshape((hparams['output_seq_len'], 1))(inp)

    for _ in range(hparams['num_critic_blocks']):
        x = block(x)

    c = tf.keras.layers.Conv1D(32, 3, padding='same')
    x = tfa.layers.SpectralNormalization(c, power_iterations=10)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(50)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Dense(15)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    out = tf.keras.layers.Dense(1)(x)

    model = tf.keras.models.Model(inp, out, name='critic')

    return model


def create_lstm_critic_complex(hparams):

    ls = hparams['latent_size']

    def block(x):
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(ls, return_sequences=True))(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        return x

    inp = tf.keras.layers.Input((hparams['output_seq_len'],))
    x = tf.keras.layers.Reshape((hparams['output_seq_len'], 1))(inp)

    for _ in range(hparams['num_critic_blocks']):
        x = block(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(ls, return_sequences=True))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(50)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Dense(15)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    out = tf.keras.layers.Dense(1)(x)

    model = tf.keras.models.Model(inp, out, name='critic')

    return model
