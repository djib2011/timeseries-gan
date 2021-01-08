import tensorflow as tf


def bidirectional_2_layer_bn(hparams):
    s = hparams['base_layer_size']
    inp = tf.keras.layers.Input(shape=(hparams['input_seq_length'], 1))
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s, return_sequences=True))(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(s * 2, return_sequences=True))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(hparams['output_seq_length'] * s // 2, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Reshape((hparams['output_seq_length'], s // 2))(x)
    out = tf.keras.layers.LSTM(1, return_sequences=True)(x)
    model = tf.keras.models.Model(inp, out)
    model.compile(loss='mae', optimizer='adam', metrics=['mae', 'mse'])
    return model


model_dict = {'forecaster_2layer_bn': bidirectional_2_layer_bn}

if __name__ == '__main__':

    hparams = {'input_seq_length': 24, 'base_layer_size': 32, 'output_seq_length': 6}

    model = bidirectional_2_layer_bn(hparams)
    model.summary()
