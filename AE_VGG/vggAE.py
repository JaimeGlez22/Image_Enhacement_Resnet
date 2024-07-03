from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

class VGGAE:

    def __init__(self, input_shape = (32, 32, 1),
                 block_filters = [64, 128, 256, 512, 512], block_layers=[2,2,3,3,3],
                 output_channels=1, vgg_decoder=True, kernel_regularizer=None):
        self.input_shape = input_shape
        self.block_filters = block_filters
        self.block_layers = block_layers
        self.output_channels = output_channels
        self.use_vgg_decoder = vgg_decoder
        self.model = self.build_model()
    
    
    def VGG_encoder(self, input_shape):
        X_input = Input(input_shape[1:])

        X = X_input

        stage = 1
        for i in range(len(self.block_filters)):
            filter = self.block_filters[i]
            for layer in range(self.block_layers[i]):
                X = Conv2D(filter, (3, 3), activation='relu', padding='same',
                            name='conv_' + str(stage) + chr(ord('a')+layer),
                            kernel_initializer = glorot_uniform(seed=0))(X)
            X = MaxPooling2D((2, 2), strides=(2,2), name='max_pool_' + str(stage))(X)
            stage += 1

        model = Model(inputs = X_input, outputs = X, name='VGG_encoder')
        return model
    
    def VGG_decoder(self, encoder_output):
        X_input = Input(encoder_output[1:])

        X = X_input

        # bottleneck layers
        stage = 6
        for i in range(len(self.block_filters)-1, -1, -1):
            filter = self.block_filters[i]
            
            if i == 0:                
                X = Conv2DTranspose(filter, (3, 3), activation='relu', padding='same',strides=(2, 2),
                                    name='inv_conv_' + str(stage) + "a",
                                    kernel_initializer = glorot_uniform(seed=0))(X)
                X = Conv2DTranspose(self.output_channels, (3, 3), activation='sigmoid', padding='same',
                                    name='inv_conv_' + str(stage) + "b",
                                    kernel_initializer = glorot_uniform(seed=0))(X)
            else:
                X = Conv2DTranspose(filter, (3, 3), activation='relu', padding='same',strides=(2, 2),
                                     name='inv_conv_' + str(stage) + 'a',
                                     kernel_initializer = glorot_uniform(seed=0))(X)
                for layer in range(self.block_layers[i]-1):
                    X = Conv2DTranspose(filter, (3, 3), activation='relu', padding='same',
                                        name='inv_conv_' + str(stage) + chr(ord('a')+layer+1),
                                        kernel_initializer = glorot_uniform(seed=0))(X)
            
            stage += 1
        

        model = Model(inputs = X_input, outputs = X, name='VGG_decoder')
        return model
    
    def decoder(self, encoder_output):
        X_input = Input(encoder_output[1:])

        X = X_input

        X = Conv2DTranspose(512, (3,3), strides = (2, 2), name = 'inv_conv5', kernel_initializer = glorot_uniform(seed=0), padding="same")(X)
        X = BatchNormalization(axis = 3, name = 'inv_bn_conv5')(X)
        X = Activation('relu')(X)

        X = Conv2DTranspose(256, (3,3), strides = (2, 2), name = 'inv_conv4', kernel_initializer = glorot_uniform(seed=0), padding="same")(X)
        X = BatchNormalization(axis = 3, name = 'inv_bn_conv4')(X)
        X = Activation('relu')(X)

        X = Conv2DTranspose(128, (3,3), strides = (2, 2), name = 'inv_conv3', kernel_initializer = glorot_uniform(seed=0), padding="same")(X)
        X = BatchNormalization(axis = 3, name = 'inv_bn_conv3')(X)
        X = Activation('relu')(X)

        X = Conv2DTranspose(64, (3,3), strides = (2, 2), name = 'inv_conv2', kernel_initializer = glorot_uniform(seed=0), padding="same")(X)
        X = BatchNormalization(axis = 3, name = 'inv_bn_conv2')(X)
        X = Activation('relu')(X)

        X = Conv2DTranspose(self.output_channels, (3,3), strides = (2, 2), name = 'inv_conv1', kernel_initializer = glorot_uniform(seed=0), padding="same")(X)
        X = BatchNormalization(axis = 3, name = 'inv_bn_conv1')(X)
        X = Activation('sigmoid')(X)

        model = Model(inputs = X_input, outputs = X, name='Decoder')
        return model

    
    def build_encoder(self):
        input = Input(self.input_shape)
        encoder = self.VGG_encoder(input.shape)(input)
        model = Model(inputs = input, outputs = encoder, name='VGGEncoder')
        return model
    
    def build_model(self):
        input = Input(self.input_shape)
        encoder = self.VGG_encoder(input.shape)(input)
        if self.use_vgg_decoder:
            decoder = self.VGG_decoder(encoder.shape)(encoder)
        else:
            decoder = self.decoder(encoder.shape)(encoder)
        model = Model(inputs = input, outputs = decoder, name='VGGAE')
        return model
    
    def encoder_summary(self):
        input = Input(self.input_shape)
        encoder = self.VGG_encoder(input.shape)
        return encoder.summary()
    
    def decoder_summary(self):
        input = Input(self.input_shape)
        encoder = self.VGG_encoder(input.shape)
        decoder = self.VGG_decoder(encoder.output.shape)
        return decoder.summary()
    
    def summary(self):
        return self.model.summary()
    
    def compile(self, optimizer='adam', loss='mean_squared_error'):
        self.model.compile(optimizer=optimizer, loss=loss)

    def fit(self, x, y, batch_size, epochs, callbacks, validation_data=None, shuffle=True, validation_split=0.1):
        if validation_data is None:
            return self.model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=callbacks, shuffle=shuffle)
        else: 
            return self.model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_data=validation_data, callbacks=callbacks, shuffle=shuffle)
    
    def predict(self, x):
        return self.model.predict(x)
    
    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = load_model(path)
        return self.model
    
    def evaluate(self, y_test, y_pred, channel_axis=None):
        s = ssim(y_test, y_pred, data_range=y_test.max() - y_test.min(), channel_axis=channel_axis)
        p = psnr(y_test, y_pred, data_range=y_pred.max() - y_pred.min())

        print(f'SSIM: {s}')
        print(f'PSNR: {p}')

    


        
