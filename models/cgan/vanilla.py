import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import sys

sys.path.append(os.getcwd())
import datasets
from models.cgan import generators, discriminators


class cGAN(tf.keras.Model):
    """
    https://github.com/timsainb/tensorflow2-generative-models/blob/master/2.0-GAN-fashion-mnist.ipynb
    """

    def __init__(self, **kwargs):
        super(cGAN, self).__init__()
        self.__dict__.update(kwargs)

    def generate(self, z, condition):
        return self.gen((z, condition))

    def discriminate(self, x, condition):
        return self.disc((x, condition))

    def generate_n_samples(self, condition):
        z = tf.random.normal([condition.shape[0], self.latent_size, condition.shape[1]])
        return self.generate(z, condition)

    def compute_loss(self, x, condition):
        """ passes through the network and computes loss
        """

        # generating noise from a uniform distribution
        z_samp = tf.random.normal([tf.shape(x)[0], self.latent_size])

        # run noise through generator
        x_gen = self.generate(z_samp, condition)
        # discriminate x and x_gen
        logits_x = self.discriminate(x, condition)
        logits_x_gen = self.discriminate(x_gen, condition)


        ### losses
        # losses of real with label "1"
        disc_real_loss = self.loss(logits=logits_x, is_real=True)

        # losses of fake with label "0"
        disc_fake_loss = self.loss(logits=logits_x_gen, is_real=False)
        disc_loss = disc_fake_loss + disc_real_loss

        # losses of fake with label "1"
        gen_loss = self.loss(logits=logits_x_gen, is_real=True)

        return disc_loss, gen_loss

    def compute_gradients(self, x, condition):
        """ passes through the network and computes loss
        """

        ### pass through network
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            disc_loss, gen_loss = self.compute_loss(x, condition)

        # compute gradients
        gen_gradients = gen_tape.gradient(gen_loss, self.gen.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.disc.trainable_variables)

        return gen_gradients, disc_gradients

    def apply_gradients(self, gen_gradients, disc_gradients):

        self.gen_optimizer.apply_gradients(zip(gen_gradients, self.gen.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(disc_gradients, self.disc.trainable_variables))

    @staticmethod
    def loss(logits, is_real=True):
        """
        Computes standard gan loss between logits and labels
        """
        if is_real:
            labels = tf.ones_like(logits)
        else:
            labels = tf.zeros_like(logits)

        return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

    def train_single_batch(self, x, condition):
        gen_gradients, disc_gradients = self.compute_gradients(x, condition)
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

            for i, (x, condition) in enumerate(train_generator):
                self.train_single_batch(x, condition)
                if i >= train_steps:
                    break

            if save_weights:
                self.save_models(result_dir, epoch)

            g_loss, d_loss = [], []
            for i, (x, condition) in enumerate(valid_generator):
                g, d = self.compute_loss(x, condition)
                g_loss.append(g.numpy().mean())
                d_loss.append(d.numpy().mean())
                if i >= valid_steps:
                    break

            g_losses.append(g_loss)
            d_losses.append(d_loss)

        return g_losses, d_losses


def make_vanilla_lstm_large(hparams):

    generator = generators.create_lstm_generator_large(hparams)

    discriminator = discriminators.create_lstm_discriminator_large(hparams)

    gen_optimizer = tf.keras.optimizers.Adam(0.001, beta_1=0.5)
    disc_optimizer = tf.keras.optimizers.RMSprop(0.005)

    cgan = cGAN(gen=generator, disc=discriminator, gen_optimizer=gen_optimizer, disc_optimizer=disc_optimizer,
                latent_size=hparams['latent_size'])

    return cgan


if __name__ == '__main__':
    hparams = {'latent_size': 5, 'output_seq_len': 24, 'condition_size': 10}
    batch_size = 512

    generator = generators.create_lstm_generator_large(hparams)
    generator.summary()

    discriminator = discriminators.create_lstm_discriminator_large(hparams)
    discriminator.summary()

    gen_optimizer = tf.keras.optimizers.Adam(0.001, beta_1=0.5)
    disc_optimizer = tf.keras.optimizers.RMSprop(0.005)

    # model
    gan = cGAN(gen=generator, disc=discriminator, gen_optimizer=gen_optimizer, disc_optimizer=disc_optimizer,
               latent_size=5)

    train_path = 'data/yearly_24_nw_train.h5'
    test_path = 'data/yearly_24_nw_test.h5'
    train_feats_path = 'data/yearly_24_nw_feats_train.h5'
    test_feats_path = 'data/yearly_24_nw_feats_test.h5'

    train_gen = datasets.cgan_generator(train_path, train_feats_path, batch_size=batch_size, shuffle=True)
    test_gen = datasets.cgan_generator(test_path, test_feats_path, batch_size=batch_size, shuffle=True)

    g_losses, d_losses = gan.train(train_gen, test_gen,
                                   train_steps=len(train_gen) // batch_size + 1,
                                   valid_steps=len(train_gen) // batch_size + 1,
                                   result_dir='/tmp/gan_vanilla/',
                                   save_weights=False)
