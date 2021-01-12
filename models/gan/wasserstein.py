import tensorflow as tf
import tensorflow_addons as tfa
import sys
import os
from tqdm import tqdm

sys.path.append(os.getcwd())
import datasets
from models.gan import generators, discriminators


class WGAN(tf.keras.Model):
    """[summary]
    I used github/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Tensorflow-2/ as a reference on this.

    Extends:
        tf.keras.Model
    """

    def __init__(self, **kwargs):
        super(WGAN, self).__init__()
        self.__dict__.update(kwargs)

    def generate(self, z):
        return self.gen(z)

    def discriminate(self, x):
        return self.disc(x)

    def compute_loss(self, x):
        """
        passes through the network and computes loss
        """
        ### pass through network
        # generating noise from a uniform distribution

        z_samp = tf.random.normal([tf.shape(x)[0], self.latent_size])

        # run noise through generator
        x_gen = self.generate(z_samp)
        # discriminate x and x_gen
        logits_x = self.discriminate(x)
        logits_x_gen = self.discriminate(x_gen)

        # gradient penalty
        d_regularizer = self.gradient_penalty(x, x_gen)
        ### losses
        disc_loss = tf.reduce_mean(logits_x) - tf.reduce_mean(logits_x_gen) + d_regularizer * self.gradient_penalty_weight

        # losses of fake with label "1"
        gen_loss = tf.reduce_mean(logits_x_gen)

        return disc_loss, gen_loss

    def compute_gradients(self, x):
        """ passes through the network and computes loss
        """
        ### pass through network
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            disc_loss, gen_loss = self.compute_loss(x)

        # compute gradients
        gen_gradients = gen_tape.gradient(gen_loss, self.gen.trainable_variables)

        disc_gradients = disc_tape.gradient(disc_loss, self.disc.trainable_variables)

        return gen_gradients, disc_gradients

    def apply_gradients(self, gen_gradients, disc_gradients):
        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.gen.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.disc.trainable_variables))

    def gradient_penalty(self, x, x_gen):
        epsilon = tf.random.uniform((tf.shape(x)[0], 1, 1), 0.0, 1.0)
        x_hat = epsilon * tf.cast(x, tf.float32) + (1 - epsilon) * tf.expand_dims(x_gen, -1)
        with tf.GradientTape() as t:
            t.watch(x_hat)
            d_hat = self.discriminate(x_hat)
        gradients = t.gradient(d_hat, x_hat)
        ddx = tf.sqrt(tf.reduce_sum(gradients ** 2, axis=[1, 2]))
        d_regularizer = tf.reduce_mean((ddx - 1.0) ** 2)
        return d_regularizer

    def train_single_batch(self, x):
        gen_gradients, disc_gradients = self.compute_gradients(x)
        self.apply_gradients(gen_gradients, disc_gradients)

    def save_models(self, result_dir, epoch):
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        self.gen.save(result_dir + 'generator_epoch_{}.h5'.format(epoch))
        self.disc.save(result_dir + 'discriminator_epoch_{}.h5'.format(epoch))

    def load_models(self, gen_path, disc_path):
        self.gen = tf.keras.models.load_model(gen_path)
        self.disc = tf.keras.models.load_model(disc_path)

    # @tf.function
    def train(self, train_generator, valid_generator, train_steps, valid_steps, epochs=50,
              result_dir='results/vanilla/', save_weights=False):

        g_losses, d_losses = [], []

        for epoch in tqdm(range(epochs)):

            for i, x in enumerate(train_generator):
                self.train_single_batch(x)
                if i >= train_steps:
                    break

            if save_weights:
                self.save_models(result_dir, epoch)

            g_loss, d_loss = [], []
            for i, x in enumerate(valid_generator):
                g, d = self.compute_loss(x)
                g_loss.append(g.numpy().mean())
                d_loss.append(d.numpy().mean())
                if i >= valid_steps:
                    break

            g_losses.append(g_loss)
            d_losses.append(d_loss)

        return g_losses, d_losses


def make_wgan_lstm_large(hparams):

    generator = generators.create_lstm_generator_large(hparams)

    discriminator = discriminators.create_lstm_discriminator_large(hparams)

    gen_optimizer = tf.keras.optimizers.Adam(0.001, beta_1=0.5)
    disc_optimizer = tf.keras.optimizers.RMSprop(0.005)

    gan = WGAN(gen=generator, disc=discriminator, gen_optimizer=gen_optimizer, disc_optimizer=disc_optimizer,
              latent_size=hparams['latent_size'], gradient_penalty_weight=hparams['gp_weight'])

    return gan


if __name__ == '__main__':

    hparams = {'latent_size': 5, 'output_seq_len': 24}

    generator = create_lstm_generator_large(hparams)
    generator.summary()

    discriminator = create_lstm_critic_large(hparams)
    discriminator.summary()

    # optimizers
    gen_optimizer = tf.keras.optimizers.Adam(0.001, beta_1=0.5)
    disc_optimizer = tf.keras.optimizers.RMSprop(0.005)

    # model
    gan = WGAN(gen=generator, disc=discriminator, gen_optimizer=gen_optimizer, disc_optimizer=disc_optimizer,
               gradient_penalty_weight=10., latent_size=5)

    train_path = 'data/yearly_24_nw_train.h5'
    test_path = 'data/yearly_24_nw_test.h5'

    batch_size = 512

    train_gen = datasets.gan_generator(train_path, batch_size=1024, shuffle=True)
    test_gen = datasets.gan_generator(test_path, batch_size=1024, shuffle=True)

    g_losses, d_losses = gan.train(train_gen, test_gen,
                                   train_steps=len(train_gen) // batch_size + 1,
                                   valid_steps=len(train_gen) // batch_size + 1,
                                   result_dir='/tmp/wgan/',
                                   save_weights=False)
