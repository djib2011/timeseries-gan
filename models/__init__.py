from models import classification, forecast, gan

model_dict = classification.model_dict
model_dict.update(forecast.model_dict)
model_dict.update(gan.model_dict)

def get_model(name):
    return model_dict[name]
