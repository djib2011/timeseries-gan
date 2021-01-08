from models.gan import wasserstein, vanilla

model_dict = {'vanilla_conv_gan': vanilla.make_vanilla_conv,
              'vanilla_lstm_small_gan': vanilla.make_vanilla_lstm_small,
              'vanilla_lstm_large_gan': vanilla.make_vanilla_lstm_large}
