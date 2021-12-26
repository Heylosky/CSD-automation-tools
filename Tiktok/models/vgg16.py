from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import GlobalAveragePooling2D
import tensorflow as tf

class vgg16:
     @staticmethod
     def build(IMAGE_DIMS, classes, finalAct="softmax"):
          base_model = tf.keras.applications.VGG16(input_shape=IMAGE_DIMS,
                                             include_top=False,
                                             weights='imagenet')
          
          model = Sequential()
          base_model.trainable = False
          model.add(base_model)
          model.add(GlobalAveragePooling2D())
          
          model.add(Dense(4096))
          model.add(Activation("relu"))
          model.add(BatchNormalization())
          model.add(Dropout(0.5))
          
          model.add(Dense(4096))
          model.add(Activation("relu"))
          model.add(BatchNormalization())
          model.add(Dropout(0.5))
          
          model.add(Dense(classes))
          model.add(Activation(finalAct))
          
          return model