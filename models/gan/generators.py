import tensorflow as tf
import tensorflow_addons as tfa


def create_conv_generator_simple(hparams):

    inp = tf.keras.layers.Input((hparams['latent_size'],))
    x = tf.keras.layers.Dense(hparams['latent_size'], activation='relu')(inp)
    x = tf.keras.layers.Reshape((hparams['latent_size'], 1))(x)
    x = tf.keras.layers.Conv1DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters=1, kernel_size=3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(hparams['output_seq_len'])(x)

    model = tf.keras.models.Model(inp, out, name='generator')
    return model


def create_lstm_generator_simple(hparams):
    ls = hparams['latent_size']
    inp = tf.keras.layers.Input((ls,))
    x = tf.keras.layers.Dense(ls, activation='relu')(inp)
    x = tf.keras.layers.Reshape((ls, 1))(x)
    x = tf.keras.layers.LSTM(ls*8, return_sequences=True)(x)
    x = tf.keras.layers.UpSampling1D()(x)
    x = tf.keras.layers.LSTM(ls*4, return_sequences=True)(x)
    x = tf.keras.layers.UpSampling1D()(x)
    x = tf.keras.layers.LSTM(ls*2, return_sequences=True)(x)
    x = tf.keras.layers.UpSampling1D()(x)
    x = tf.keras.layers.LSTM(ls, return_sequences=True)(x)
    x = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(hparams['output_seq_len'])(x)
    model = tf.keras.models.Model(inp, out, name='generator')
    return model


def create_lstm_generator_large(hparams):
    ls = hparams['latent_size']
    inp = tf.keras.layers.Input((ls,))
    x = tf.keras.layers.Dense(ls, activation='relu')(inp)
    x = tf.keras.layers.Reshape((ls, 1))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(ls*16, return_sequences=True))(x)
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


def create_conv_generator_complex(hparams):

    def block(x):
        c = tf.keras.layers.Conv1D(32, 3, padding='same')
        x = tfa.layers.SpectralNormalization(c, power_iterations=10)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = tf.keras.layers.UpSampling1D(size=2)(x)
        return x

    inp = tf.keras.layers.Input((hparams['latent_size'],))
    x = tf.keras.layers.Dense(hparams['latent_size'])(inp)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = tf.keras.layers.Reshape((hparams['latent_size'], 1))(x)

    for _ in range(hparams['num_generator_blocks']):
        x = block(x)

    c = tf.keras.layers.Conv1D(32, 3, padding='same')
    x = tfa.layers.SpectralNormalization(c, power_iterations=10)(x)
    c = tf.keras.layers.Conv1D(1, 3, padding='same')
    x = tfa.layers.SpectralNormalization(c, power_iterations=10)(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    x = tf.keras.layers.Flatten()(x)
    out = tf.keras.layers.Dense(hparams['output_seq_len'])(x)

    model = tf.keras.models.Model(inp, out, name='generator')

    return model
