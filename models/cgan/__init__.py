from models.cgan import generators, discriminators, vanilla

model_dict = {'vanilla_lstm_large_cgan': vanilla.make_vanilla_lstm_large,
              'vanilla_lstm_large_cgan_2': vanilla.make_vanilla_lstm_large}
