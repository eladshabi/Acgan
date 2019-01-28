from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, InputSpec
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import set_floatx,set_epsilon

from tensorflow.keras.layers import BatchNormalization, Layer
from tensorflow.keras.initializers import Initializer
from tensorflow.keras import initializers, regularizers,constraints

from tensorflow.keras import backend as K


class Ones32(Initializer):
    def __call__(self, shape, dtype=None):
        return K.constant(1, shape=shape, dtype='float32')


class Zeros32(Initializer):
    def __call__(self, shape, dtype=None):
        return K.constant(0, shape=shape, dtype='float32')


class BatchNormalizationF16(BatchNormalization):

    # class creator with same params as a regular batch normalization
    # uses the float32 initializers as default
    def __init__(self,
                 beta_initializer=Zeros32(),
                 gamma_initializer=Ones32(),
                 moving_mean_initializer=Zeros32(),
                 moving_variance_initializer=Ones32(),
                 **kwargs):

        super(BatchNormalizationF16, self).__init__(
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
            **kwargs)

    # method that creates and initializes the weights - forcing float32
    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                                                        'input tensor should have a defined dimension '
                                                        'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        # forcing float32 here
        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         dtype='float32',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            # forcing float32 here
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        dtype='float32',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None

        # forcing float32 here
        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            dtype='float32',
            initializer=self.moving_mean_initializer,
            trainable=False)

        # forcing float32 here
        self.moving_variance = self.add_weight(
            shape=shape,
            name='moving_variance',
            dtype='float32',
            initializer=self.moving_variance_initializer,
            trainable=False)
        self.built = True

    # here we need to cast to and back from float32
    def call(self, inputs, training=None):
        inputs = K.cast(inputs, 'float32')
        result = super(BatchNormalizationF16, self).call(inputs, training)
        return K.cast(result, K.floatx())

class ACGAN():
    def __init__(self,rows,cols,channels,classes,latent,tpu = False):

        if tpu:
            set_floatx('float16')
            set_epsilon(1e-4)

        # Input shape
        self.img_rows = rows
        self.img_cols = cols
        self.channels = channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.num_of_classes = classes

        # size of the vector to fid the generator (z)
        self.latent_dim = latent

        optimizer = Adam(0.0002, 0.5)

        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses,
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid, target_label = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([noise, label], [valid, target_label])
        self.combined.compile(loss=losses,
                              optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(BatchNormalizationF16(momentum=0.8))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(BatchNormalizationF16(momentum=0.8))

        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(BatchNormalizationF16(momentum=0.8))
        model.add(Conv2D(self.channels, kernel_size=3, padding='same'))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_of_classes, 100)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(BatchNormalizationF16(momentum=0.8))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        #model.add(BatchNormalization(momentum=0.8))
        model.add(BatchNormalizationF16(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.summary()

        img = Input(shape=self.img_shape)

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_of_classes, activation="softmax")(features)

        return Model(img, [validity, label])


