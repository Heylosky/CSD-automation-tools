from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense

class likevgg16:
     @staticmethod
     def build(IMAGE_DIMS, classes, finalAct="softmax"):
          model = Sequential()
          chanDim = -1
          
          model.add(Conv2D(64, (3, 3), padding="same", input_shape=IMAGE_DIMS))
          model.add(Activation("relu"))
          model.add(BatchNormalization(axis=chanDim))
          model.add(Conv2D(64, (3, 3), padding="same"))
          model.add(Activation("relu"))
          model.add(BatchNormalization(axis=chanDim)) 
          model.add(MaxPooling2D(pool_size=(2, 2))) 
          model.add(Dropout(0.25))
          
          model.add(Conv2D(128, (3, 3), padding="same"))
          model.add(Activation("relu"))
          model.add(BatchNormalization(axis=chanDim))
          model.add(Conv2D(128, (3, 3), padding="same"))
          model.add(Activation("relu"))
          model.add(BatchNormalization(axis=chanDim)) 
          model.add(MaxPooling2D(pool_size=(2, 2))) 
          model.add(Dropout(0.25))
          
          model.add(Conv2D(256, (3, 3), padding="same"))
          model.add(Activation("relu"))
          model.add(BatchNormalization(axis=chanDim))
          model.add(Conv2D(256, (3, 3), padding="same"))
          model.add(Activation("relu"))
          model.add(BatchNormalization(axis=chanDim))
          model.add(Conv2D(256, (3, 3), padding="same"))
          model.add(Activation("relu"))
          model.add(BatchNormalization(axis=chanDim))
          model.add(MaxPooling2D(pool_size=(2, 2)))
          model.add(Dropout(0.25))
          
          model.add(Conv2D(512, (3, 3), padding="same"))
          model.add(Activation("relu"))
          model.add(BatchNormalization(axis=chanDim))
          model.add(Conv2D(512, (3, 3), padding="same"))
          model.add(Activation("relu"))
          model.add(BatchNormalization(axis=chanDim))
          model.add(Conv2D(512, (3, 3), padding="same"))
          model.add(Activation("relu"))
          model.add(BatchNormalization(axis=chanDim))
          model.add(MaxPooling2D(pool_size=(2, 2)))
          model.add(Dropout(0.25))
          
          model.add(Conv2D(512, (3, 3), padding="same"))
          model.add(Activation("relu"))
          model.add(BatchNormalization(axis=chanDim))
          model.add(Conv2D(512, (3, 3), padding="same"))
          model.add(Activation("relu"))
          model.add(BatchNormalization(axis=chanDim))
          model.add(Conv2D(512, (3, 3), padding="same"))
          model.add(Activation("relu"))
          model.add(BatchNormalization(axis=chanDim))
          model.add(MaxPooling2D(pool_size=(2, 2)))
          model.add(Dropout(0.25))
          
          model.add(Flatten())
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