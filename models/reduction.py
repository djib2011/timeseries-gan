import tensorflow as tf


def bidirectional_2_layer_bn(hparams):
    s = hparams['base_layer_size']

    # Input
    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))

    # Encoder
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s * 2, return_sequences=True))(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Bottleneck
    x = tf.keras.layers.Flatten()(x)
    encoded = tf.keras.layers.Dense(2)(x)
    x = tf.keras.layers.BatchNormalization()(encoded)
    x = tf.keras.layers.Dense(hparams['input_seq_length'] * s)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Decoder
    x = tf.keras.layers.Reshape((hparams['input_seq_length'], s))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s * 2, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # Output
    out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1, return_sequences=True))(x)

    encoder = tf.keras.models.Model(inp, encoded)
    autoencoder = tf.keras.models.Model(inp, out)

    autoencoder.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
    return encoder, autoencoder


model_dict = {'autoencoder_2layer_bn': bidirectional_2_layer_bn}

if __name__ == '__main__':

    hparams = {'input_seq_length': 24, 'base_layer_size': 32}

    model = bidirectional_2_layer_bn(hparams)
    model.summary()
