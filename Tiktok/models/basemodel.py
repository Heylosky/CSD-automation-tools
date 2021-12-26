from keras.models import Model, Input
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization, Flatten, Dense, Add, ZeroPadding2D, AveragePooling2D
from keras.initializers import glorot_uniform

class baseModel:
     @staticmethod
     def initialize_inputs(IMAGE_DIMS):
          model_input = Input(shape=IMAGE_DIMS)
          return model_input
     
     def base_smallervggnet(inputs, classes, finalAct="softmax"):
          x = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
          x = BatchNormalization(axis=-1)(x)
          x = MaxPooling2D((3,3), strides=(2,2))(x)
          x = Dropout(0.25)(x)
          
          x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
          x = BatchNormalization(axis=-1)(x)
          x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
          x = BatchNormalization(axis=-1)(x)
          x = MaxPooling2D((2,2), strides=(2,2))(x)
          x = Dropout(0.25)(x)
          
          x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
          x = BatchNormalization(axis=-1)(x)
          x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
          x = BatchNormalization(axis=-1)(x)
          x = MaxPooling2D((2,2), strides=(2,2))(x)
          x = Dropout(0.25)(x)
          
          x = Flatten()(x)
          x = Dense(1024, activation='relu')(x)
          x = BatchNormalization()(x)
          x = Dropout(0.5)(x)
          
          x = Dense(classes, activation=(finalAct))(x)
          
          model = Model(inputs, x)
          
          return model
     
     def base_likevgg(inputs, classes, finalAct="softmax"):
          x = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
          x = BatchNormalization(axis=-1)(x)
          x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
          x = BatchNormalization(axis=-1)(x)
          x = MaxPooling2D((2,2), strides=(2,2))(x)
          x = Dropout(0.25)(x)
          
          x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
          x = BatchNormalization(axis=-1)(x)
          x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
          x = BatchNormalization(axis=-1)(x)
          x = MaxPooling2D((2,2), strides=(2,2))(x)
          x = Dropout(0.25)(x)
          
          x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
          x = BatchNormalization(axis=-1)(x)
          x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
          x = BatchNormalization(axis=-1)(x)
          x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
          x = BatchNormalization(axis=-1)(x)
          x = MaxPooling2D((2,2), strides=(2,2))(x)
          x = Dropout(0.25)(x)
          
          x = Conv2D(512, (3,3), activation='relu', padding='same')(x)
          x = BatchNormalization(axis=-1)(x)
          x = Conv2D(512, (3,3), activation='relu', padding='same')(x)
          x = BatchNormalization(axis=-1)(x)
          x = Conv2D(512, (3,3), activation='relu', padding='same')(x)
          x = BatchNormalization(axis=-1)(x)
          x = MaxPooling2D((2,2), strides=(2,2))(x)
          x = Dropout(0.25)(x)
          
          x = Conv2D(512, (3,3), activation='relu', padding='same')(x)
          x = BatchNormalization(axis=-1)(x)
          x = Conv2D(512, (3,3), activation='relu', padding='same')(x)
          x = BatchNormalization(axis=-1)(x)
          x = Conv2D(512, (3,3), activation='relu', padding='same')(x)
          x = BatchNormalization(axis=-1)(x)
          x = MaxPooling2D((2,2), strides=(2,2))(x)
          x = Dropout(0.25)(x)
          
          x = Flatten()(x)
          x = Dense(4096, activation='relu')(x)
          x = BatchNormalization()(x)
          x = Dropout(0.5)(x)
          
          x = Dense(4096, activation='relu')(x)
          x = BatchNormalization()(x)
          x = Dropout(0.5)(x)
          
          x = Dense(1000, activation='relu')(x)
          x = BatchNormalization()(x)
          x = Dropout(0.5)(x)
          
          x = Dense(classes, activation=(finalAct))(x)
          
          model = Model(inputs, x)
          
          return model
     
     def indentity_block(X, f, filters, stage, block):
          conv_name_base = 'res' + str(stage) + block + '_branch'
          bn_name_base = 'bn' + str(stage) + block + '_branch'
          F1, F2, F3 = filters
          
          X_shortcut = X
          
          X = Conv2D(filters = F1, kernel_size = (1,1), strides = (1,1), padding = 'valid',
                     name = conv_name_base + '2a' , kernel_initializer = glorot_uniform(seed = 0))(X)
          X = BatchNormalization(axis = 3, name = bn_name_base +'2a')(X)
          X = Activation('relu')(X)
          
          X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same', 
                     name = conv_name_base + '2b' , kernel_initializer = glorot_uniform(seed = 0))(X)
          X = BatchNormalization(axis = 3, name = bn_name_base +'2b')(X)
          X = Activation('relu')(X)
          
          X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), padding = 'valid', 
                     name = conv_name_base + '2c' , kernel_initializer = glorot_uniform(seed = 0))(X)
          X = BatchNormalization(axis = 3, name = bn_name_base +'2c')(X)
          
          X = Add()([X, X_shortcut])
          X = Activation('relu')(X)
          
          return X
          
     def convolution_block(X, f, filters, stage, block, s=2):
          conv_name_base = 'res' + str(stage) + block + '_branch'
          bn_name_base = 'bn' + str(stage) + block + '_branch'
          F1, F2, F3 = filters
          
          X_shortcut = X
          
          X = Conv2D(filters = F1, kernel_size = (1,1), strides = (s,s),
                     name = conv_name_base + '2a' , kernel_initializer = glorot_uniform(seed = 0))(X)
          X = BatchNormalization(axis = 3, name = bn_name_base +'2a')(X)
          X = Activation('relu')(X)
          
          X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same',
                     name = conv_name_base + '2b' , kernel_initializer = glorot_uniform(seed = 0))(X)
          X = BatchNormalization(axis = 3, name = bn_name_base +'2b')(X)
          X = Activation('relu')(X)
          
          X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1),
                     name = conv_name_base + '2c' , kernel_initializer = glorot_uniform(seed = 0))(X)
          X = BatchNormalization(axis = 3, name = bn_name_base +'2c')(X)
          
          X_shortcut = Conv2D(F3, (1,1), strides = (s,s), name = conv_name_base + '1',
                              kernel_initializer = glorot_uniform(seed = 0))(X_shortcut)
          X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)
          
          X = Add()([X, X_shortcut])
          X = Activation('relu')(X)
          
          return X
     
     def base_restnet50(inputs, classes, finalAct="softmax"):
          X = ZeroPadding2D((3,3))(inputs)
          
          X = Conv2D(64, (7,7), strides = (2,2), name = 'conv1', kernel_initializer=glorot_uniform(seed=0))(X)
          X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
          X = Activation('relu')(X)
          X = MaxPooling2D((3,3), strides=(2,2))(X)
          
          X = baseModel.convolution_block(X, f=3, filters=[64,64,256], stage=2, block='a', s=1)
          X = baseModel.indentity_block(X, 3, [64,64,256], stage=2, block='b')
          X = baseModel.indentity_block(X, 3, [64,64,256], stage=2, block='c')
          
          X = baseModel.convolution_block(X, f=3, filters=[128,128,512], stage=3, block='a', s=2)
          X = baseModel.indentity_block(X, 3, [128,128,512], stage=3, block='b')
          X = baseModel.indentity_block(X, 3, [128,128,512], stage=3, block='c')
          X = baseModel.indentity_block(X, 3, [128,128,512], stage=3, block='d')
          
          X = baseModel.convolution_block(X, f=3, filters=[256,256,1024], stage=4, block='a', s=2)
          X = baseModel.indentity_block(X, 3, [256,256,1024], stage=4, block='b')
          X = baseModel.indentity_block(X, 3, [256,256,1024], stage=4, block='c')
          X = baseModel.indentity_block(X, 3, [256,256,1024], stage=4, block='d')
          X = baseModel.indentity_block(X, 3, [256,256,1024], stage=4, block='e')
          X = baseModel.indentity_block(X, 3, [256,256,1024], stage=4, block='f')
          
          X = baseModel.convolution_block(X, f=3, filters=[512,512,2048], stage=5, block='a', s=2)
          X = baseModel.indentity_block(X, 3, [512,512,2048], stage=5, block='b')
          X = baseModel.indentity_block(X, 3, [512,512,2048], stage=5, block='c')
          
          X = AveragePooling2D((2,2), name='avg_pool')(X)
          
          X = Flatten()(X)
          X = Dense(classes, activation=(finalAct), name = 'fc'+str(classes), kernel_initializer=glorot_uniform(seed=0))(X)
          
          model = Model(inputs, outputs = X, name='Restnet50')
          
          return model