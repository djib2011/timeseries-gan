from models.gan import generators, discriminators, vanilla, wasserstein


model_dict = {'vanilla_conv_gan': vanilla.make_vanilla_conv,
              'vanilla_lstm_small_gan': vanilla.make_vanilla_lstm_small,
              'vanilla_lstm_large_gan': vanilla.make_vanilla_lstm_large,
              'wasserstein_lstm_large_gan': wasserstein.make_wgan_lstm_large}

model_dict.update({'wasserstein_{}_complex_{}_gan'.format(t, i): wasserstein.get_complex(t, i) for t in ('lstm', 'conv')
                                                                                               for i in range(1, 5)})
