from models import classification, forecast, reduction, gan

model_dict = classification.model_dict
model_dict.update(forecast.model_dict)
model_dict.update(reduction.model_dict)
model_dict.update(gan.model_dict)

def get_model(name):
    return model_dict[name]
