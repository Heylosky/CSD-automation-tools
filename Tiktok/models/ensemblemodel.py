from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Average, Dropout
from keras.models import Model, Input

class ensembleModel:
     @staticmethod 
     def ensemble_model(models, model_input):
          outputs = [model.outputs[0] for model in models]
          y = Average()(outputs)
          model = Model(model_input, y, name='ensemble')
          return model