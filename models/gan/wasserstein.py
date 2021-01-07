import tensorflow as tf
import tensorflow_addons as tfa


def create_conv_generator(hparams):

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


def create_conv_critic(hparams):

    def block(x):
        c = tf.keras.layers.Conv1D(32, 3, padding='same')
        x = tfa.layers.SpectralNormalization(c, power_iterations=10)(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        print(x.shape)
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



if __name__ == '__main__':
    hparams = {'latent_size': 5, 'num_generator_blocks': 3, 'num_critic_blocks': 2, 'output_seq_len': 24}

    generator = create_conv_generator(hparams)
    generator.summary()

    critic = create_conv_critic(hparams)
    critic.summary()
