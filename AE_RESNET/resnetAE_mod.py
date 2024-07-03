from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

class ResNetAE_mod:

    def __init__(self, input_shape = (32, 32, 1), bottleneck_num = 4,
                 bottleneck_filters = [64, 128, 256, 512], bottleneck_layers=[2,2,2,2],
                 bottleneck_strides=[1,2,2,2], bottleneck_num_layer=2, output_channels=1,
                 resnet_decoder=True):
        self.input_shape = input_shape
        self.bottleneck_num = bottleneck_num
        self.bottleneck_filters = bottleneck_filters
        self.bottleneck_layers = bottleneck_layers
        self.bottleneck_num_layer = bottleneck_num_layer
        self.bottleneck_strides = bottleneck_strides
        self.output_channels = output_channels
        self.resnet_decoder = resnet_decoder
        self.model = self.build_model()

    def _bottleneck_residual_block(self, X, filters, stage, block, reduce=False,s=2):
        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        
        # Retrieve Filters
        if len(filters) == 3:
            F1, F2, F3 = filters
            F3 = F3*4
            k1, k2, k3 = (1, 1), (3, 3), (1, 1)
        elif len(filters) == 2:
            F1, F2 = filters
            k1, k2 = (3, 3), (3, 3)

        
        # Save the input value. You'll need this later to add back to the main path. 
        X_shortcut = X

        if reduce:
            # if we are to reduce the spatial size, apply a 1x1 CONV layer to the shortcut path
            # to do that, we need both CONV layers to have similar strides 
            X = Conv2D(filters = F1, kernel_size = k1, strides = (s,s), padding = 'same',
                        name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
            X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
            X = Activation('relu')(X)

            if len(filters) == 3:
                X_shortcut = Conv2D(filters = F3, kernel_size = k1, strides = (s,s), padding = 'same',
                                    name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
                X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)
            else:
                X_shortcut = Conv2D(filters = F2, kernel_size = k1, strides = (s,s), padding = 'same',
                                    name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
                X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)
        else:
            # First component of main path
            X = Conv2D(filters = F1, kernel_size = k1, strides = (1,1), padding = 'same',
                    name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
            X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
            X = Activation('relu')(X)
        
        # Second component of main path
        X = Conv2D(filters = F2, kernel_size = k2, strides = (1,1), padding = 'same',
                name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Third component of main path
        if len(filters) == 3:
            X = Conv2D(filters = F3, kernel_size = k3, strides = (1,1), padding = 'same',
                    name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
            X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
            X = Activation('relu')(X)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X
    
    def _bottleneck_inverse_residual_block(self, X, filters, stage, block, increase=False, s=2):
        # defining name basis
        conv_name_base = 'inv_res' + str(stage) + block + '_branch'
        bn_name_base = 'inv_bn' + str(stage) + block + '_branch'
        
        # Retrieve Filters
        if len(filters) == 3:
            F1, F2, F3 = filters
            F3 = F3*4
            k1, k2, k3 = (1, 1), (3, 3), (1, 1)
        elif len(filters) == 2:
            F1, F2 = filters
            k1, k2 = (3, 3), (3, 3)
        
        # Save the input value. You'll need this later to add back to the main path. 
        X_shortcut = X

        if increase:
            # if we are to reduce the spatial size, apply a 1x1 CONV layer to the shortcut path
            # to do that, we need both CONV layers to have similar strides 
            X = Conv2DTranspose(filters = F1, kernel_size = k1, strides = (s,s), padding = 'same',
                        name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
            X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
            X = Activation('relu')(X)

            if len(filters) == 3:
                X_shortcut = Conv2DTranspose(filters = F3, kernel_size = k1, strides = (s,s), padding = 'same',
                                    name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
                X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)
            else:
                X_shortcut = Conv2DTranspose(filters = F2, kernel_size = k1, strides = (s,s), padding = 'same',
                                    name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
                X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)
        
        else:
            # First component of main path
            X = Conv2DTranspose(filters = F1, kernel_size = k1, strides = (1,1), padding = 'same',
                    name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
            X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
            X = Activation('relu')(X)
        
        # Second component of main path
        X = Conv2DTranspose(filters = F2, kernel_size = k2, strides = (1,1), padding = 'same',
                name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Third component of main path
        if len(filters) == 3:
            X = Conv2DTranspose(filters = F3, kernel_size = k3, strides = (1,1), padding = 'same',
                    name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
            X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
            X = Activation('relu')(X)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X = Add()([X, X_shortcut])
        X = Activation('relu')(X)

        return X
    
    def ResNet_encoder(self, input_shape):
        X_input = Input(input_shape[1:])

        # Stage 1
        X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', padding="same", kernel_initializer = glorot_uniform(seed=0))(X_input)
        X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(X)

        # bottleneck layers
        stage = 2
        for i in range(self.bottleneck_num):
            filter = [self.bottleneck_filters[i] for _ in range(self.bottleneck_num_layer)]
            num_layers = self.bottleneck_layers[i]
            stride = self.bottleneck_strides[i]
            X = self._bottleneck_residual_block(X, filters=filter, stage=stage, block='a', reduce=True, s=stride)
            for i in range(num_layers-1):
                # get letter by alphabetic order
                block = chr(ord('b')+i)
                X = self._bottleneck_residual_block(X, filters=filter, stage=stage, block=block)
            stage += 1

        X = AveragePooling2D((1, 1), name='avg_pool')(X)

        model = Model(inputs = X_input, outputs = X, name='ResNet_encoder')
        return model
    
    def ResNet_decoder(self, encoder_output):
        X_input = Input(encoder_output[1:])

        X = X_input

        # bottleneck layers
        stage = 5
        for i in range(self.bottleneck_num-1, -1, -1):
            filter = [self.bottleneck_filters[i] for _ in range(self.bottleneck_num_layer)]
            num_layers = self.bottleneck_layers[i]   
            X = self._bottleneck_inverse_residual_block(X, filters=filter, stage=stage, block='a', increase=True, s=2)
            for i in range(num_layers-1):
                # get letter by alphabetic order
                block = chr(ord('b')+i)
                X = self._bottleneck_inverse_residual_block(X, filters=filter, stage=stage, block=block)
            stage -= 1

        X = Conv2DTranspose(self.output_channels, (7, 7), strides = (2, 2), name = 'inv_conv5', kernel_initializer = glorot_uniform(seed=0), padding="same")(X)
        X = BatchNormalization(axis = 3, name = 'inv_bn_conv5')(X)
        X = Activation('sigmoid')(X)

        model = Model(inputs = X_input, outputs = X, name='ResNet_decoder')
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
        encoder = self.ResNet_encoder(input.shape)(input)
        model = Model(inputs = input, outputs = encoder, name='ResNetEncoder')
        return model
    
    def build_model(self):
        input = Input(self.input_shape)
        encoder = self.ResNet_encoder(input.shape)(input)
        if self.resnet_decoder:
            decoder = self.ResNet_decoder(encoder.shape)(encoder)
        else:
            decoder = self.decoder(encoder.shape)(encoder)
        
        skip_conection = layers.concatenate([input, decoder])
        output = Conv2D(self.output_channels, (3, 3), strides=1, activation='sigmoid', padding='same')(skip_conection)
        model = Model(inputs = input, outputs = output, name='ResNetAEMod')
        return model
    
    def encoder_summary(self):
        input = Input(self.input_shape)
        encoder = self.ResNet_encoder(input.shape)
        return encoder.summary()
    
    def decoder_summary(self):
        input = Input(self.input_shape)
        encoder = self.ResNet_encoder(input.shape)
        decoder = self.ResNet_decoder(encoder.output.shape)
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

    


        
