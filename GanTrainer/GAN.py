from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D,InputSpec
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, Adadelta,SGD
from tensorflow.keras.backend import set_floatx,set_epsilon

from tensorflow.keras.layers import BatchNormalization, Layer
from tensorflow.keras.initializers import Initializer
from tensorflow.keras import initializers, regularizers,constraints
from keras.backend.tensorflow_backend import tf, _regular_normalize_batch_in_training

from tensorflow.keras import backend as K
import os
import sys
import inspect
import importlib
import imp
import numpy as np
from collections import OrderedDict

class Optimizer:
    def __init__(
        self,
        name                = 'Train',
        tf_optimizer        = 'tf.train.AdamOptimizer',
        learning_rate       = 0.001,
        use_loss_scaling    = False,
        loss_scaling_init   = 64.0,
        loss_scaling_inc    = 0.0005,
        loss_scaling_dec    = 1.0,
        **kwargs):

        # Init fields.
        self.name               = name
        self.learning_rate      = tf.convert_to_tensor(learning_rate)
        self.id                 = self.name.replace('/', '.')
        self.scope              = tf.get_default_graph().unique_name(self.id)
        self.optimizer_class    = import_obj(tf_optimizer)
        self.optimizer_kwargs   = dict(kwargs)
        self.use_loss_scaling   = use_loss_scaling
        self.loss_scaling_init  = loss_scaling_init
        self.loss_scaling_inc   = loss_scaling_inc
        self.loss_scaling_dec   = loss_scaling_dec
        self._grad_shapes       = None          # [shape, ...]
        self._dev_opt           = OrderedDict() # device => optimizer
        self._dev_grads         = OrderedDict() # device => [[(grad, var), ...], ...]
        self._dev_ls_var        = OrderedDict() # device => variable (log2 of loss scaling factor)
        self._updates_applied   = False

    # Register the gradients of the given loss function with respect to the given variables.
    # Intended to be called once per GPU.
    def register_gradients(self, loss, vars):
        assert not self._updates_applied

        # Validate arguments.
        if isinstance(vars, dict):
            vars = list(vars.values()) # allow passing in Network.trainables as vars
        assert isinstance(vars, list) and len(vars) >= 1
        assert all(is_tf_expression(expr) for expr in vars + [loss])
        if self._grad_shapes is None:
            self._grad_shapes = [shape_to_list(var.shape) for var in vars]
        assert len(vars) == len(self._grad_shapes)
        assert all(shape_to_list(var.shape) == var_shape for var, var_shape in zip(vars, self._grad_shapes))
        dev = loss.device
        assert all(var.device == dev for var in vars)

        # Register device and compute gradients.
        with tf.name_scope(self.id + '_grad'), tf.device(dev):
            if dev not in self._dev_opt:
                opt_name = self.scope.replace('/', '_') + '_opt%d' % len(self._dev_opt)
                self._dev_opt[dev] = self.optimizer_class(name=opt_name, learning_rate=self.learning_rate, **self.optimizer_kwargs)
                self._dev_grads[dev] = []
            loss = self.apply_loss_scaling(tf.cast(loss, tf.float32))
            grads = self._dev_opt[dev].compute_gradients(loss, vars, gate_gradients=tf.train.Optimizer.GATE_NONE) # disable gating to reduce memory usage
            grads = [(g, v) if g is not None else (tf.zeros_like(v), v) for g, v in grads] # replace disconnected gradients with zeros
            self._dev_grads[dev].append(grads)

    # Construct training op to update the registered variables based on their gradients.
    def apply_updates(self):
        assert not self._updates_applied
        self._updates_applied = True
        devices = list(self._dev_grads.keys())
        total_grads = sum(len(grads) for grads in self._dev_grads.values())
        assert len(devices) >= 1 and total_grads >= 1
        ops = []
        with absolute_name_scope(self.scope):

            # Cast gradients to FP32 and calculate partial sum within each device.
            dev_grads = OrderedDict() # device => [(grad, var), ...]
            for dev_idx, dev in enumerate(devices):
                with tf.name_scope('ProcessGrads%d' % dev_idx), tf.device(dev):
                    sums = []
                    for gv in zip(*self._dev_grads[dev]):
                        assert all(v is gv[0][1] for g, v in gv)
                        g = [tf.cast(g, tf.float32) for g, v in gv]
                        g = g[0] if len(g) == 1 else tf.add_n(g)
                        sums.append((g, gv[0][1]))
                    dev_grads[dev] = sums

            # Sum gradients across devices.
            if len(devices) > 1:
                with tf.name_scope('SumAcrossGPUs'), tf.device(None):
                    for var_idx, grad_shape in enumerate(self._grad_shapes):
                        g = [dev_grads[dev][var_idx][0] for dev in devices]
                        if np.prod(grad_shape): # nccl does not support zero-sized tensors
                            g = tf.contrib.nccl.all_sum(g)
                        for dev, gg in zip(devices, g):
                            dev_grads[dev][var_idx] = (gg, dev_grads[dev][var_idx][1])

            # Apply updates separately on each device.
            for dev_idx, (dev, grads) in enumerate(dev_grads.items()):
                with tf.name_scope('ApplyGrads%d' % dev_idx), tf.device(dev):

                    # Scale gradients as needed.
                    if self.use_loss_scaling or total_grads > 1:
                        with tf.name_scope('Scale'):
                            coef = tf.constant(np.float32(1.0 / total_grads), name='coef')
                            coef = self.undo_loss_scaling(coef)
                            grads = [(g * coef, v) for g, v in grads]

                    # Check for overflows.
                    with tf.name_scope('CheckOverflow'):
                        grad_ok = tf.reduce_all(tf.stack([tf.reduce_all(tf.is_finite(g)) for g, v in grads]))

                    # Update weights and adjust loss scaling.
                    with tf.name_scope('UpdateWeights'):
                        opt = self._dev_opt[dev]
                        ls_var = self.get_loss_scaling_var(dev)
                        if not self.use_loss_scaling:
                            ops.append(tf.cond(grad_ok, lambda: opt.apply_gradients(grads), tf.no_op))
                        else:
                            ops.append(tf.cond(grad_ok,
                                lambda: tf.group(tf.assign_add(ls_var, self.loss_scaling_inc), opt.apply_gradients(grads)),
                                lambda: tf.group(tf.assign_sub(ls_var, self.loss_scaling_dec))))

                    # Report statistics on the last device.
                    if dev == devices[-1]:
                        with tf.name_scope('Statistics'):
                            ops.append(autosummary(self.id + '/learning_rate', self.learning_rate))
                            ops.append(autosummary(self.id + '/overflow_frequency', tf.where(grad_ok, 0, 1)))
                            if self.use_loss_scaling:
                                ops.append(autosummary(self.id + '/loss_scaling_log2', ls_var))

            # Initialize variables and group everything into a single op.
            self.reset_optimizer_state()
            init_uninited_vars(list(self._dev_ls_var.values()))
            return tf.group(*ops, name='TrainingOp')

    # Reset internal state of the underlying optimizer.
    def reset_optimizer_state(self):
        run([var.initializer for opt in self._dev_opt.values() for var in opt.variables()])

    # Get or create variable representing log2 of the current dynamic loss scaling factor.
    def get_loss_scaling_var(self, device):
        if not self.use_loss_scaling:
            return None
        if device not in self._dev_ls_var:
            with absolute_name_scope(self.scope + '/LossScalingVars'), tf.control_dependencies(None):
                self._dev_ls_var[device] = tf.Variable(np.float32(self.loss_scaling_init), name='loss_scaling_var')
        return self._dev_ls_var[device]

    # Apply dynamic loss scaling for the given expression.
    def apply_loss_scaling(self, value):
        assert is_tf_expression(value)
        if not self.use_loss_scaling:
            return value
        return value * exp2(self.get_loss_scaling_var(value.device))

    # Undo the effect of dynamic loss scaling for the given expression.
    def undo_loss_scaling(self, value):
        assert is_tf_expression(value)
        if not self.use_loss_scaling:
            return value
        return value * exp2(-self.get_loss_scaling_var(value.device))




# custom initializers to force float32
class Ones32(Initializer):
    def __call__(self, shape, dtype=None):
        return K.constant(1, shape=shape, dtype='float32')


class Zeros32(Initializer):
    def __call__(self, shape, dtype=None):
        return K.constant(0, shape=shape, dtype='float32')


class BatchNormalizationF16(Layer):

    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(BatchNormalizationF16, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = (
            initializers.get(moving_variance_initializer))
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

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

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)
        self.moving_variance = self.add_weight(
            shape=shape,
            name='moving_variance',
            initializer=self.moving_variance_initializer,
            trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        # Prepare broadcasting shape.
        ndim = len(input_shape)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        # Determines whether broadcasting is needed.
        needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])

        def normalize_inference():
            if needs_broadcasting:
                # In this case we must explicitly broadcast all parameters.
                broadcast_moving_mean = K.reshape(self.moving_mean,
                                                  broadcast_shape)
                broadcast_moving_variance = K.reshape(self.moving_variance,
                                                      broadcast_shape)
                if self.center:
                    broadcast_beta = K.reshape(self.beta, broadcast_shape)
                else:
                    broadcast_beta = None
                if self.scale:
                    broadcast_gamma = K.reshape(self.gamma,
                                                broadcast_shape)
                else:
                    broadcast_gamma = None
                return tf.nn.batch_normalization(  # K.batch_normalization(
                    inputs,
                    broadcast_moving_mean,
                    broadcast_moving_variance,
                    broadcast_beta,
                    broadcast_gamma,
                    # axis=self.axis,
                    self.epsilon)  # epsilon=self.epsilon)
            else:
                return tf.nn.batch_normalization(  # K.batch_normalization(
                    inputs,
                    self.moving_mean,
                    self.moving_variance,
                    self.beta,
                    self.gamma,
                    # axis=self.axis,
                    self.epsilon)  # epsilon=self.epsilon)

        # If the learning phase is *static* and set to inference:
        if training in {0, False}:
            return normalize_inference()

        # If the learning is either dynamic, or set to training:
        normed_training, mean, variance = _regular_normalize_batch_in_training(  # K.normalize_batch_in_training(
            inputs, self.gamma, self.beta, reduction_axes,
            epsilon=self.epsilon)

        if K.backend() != 'cntk':
            sample_size = K.prod([K.shape(inputs)[axis]
                                  for axis in reduction_axes])
            sample_size = K.cast(sample_size, dtype=K.dtype(inputs))

            # sample variance - unbiased estimator of population variance
            variance *= sample_size / (sample_size - (1.0 + self.epsilon))

        self.add_update([K.moving_average_update(self.moving_mean,
                                                 mean,
                                                 self.momentum),
                         K.moving_average_update(self.moving_variance,
                                                 variance,
                                                 self.momentum)],
                        inputs)

        # Pick the normalized form corresponding to the training phase.
        return K.in_train_phase(normed_training,
                                normalize_inference,
                                training=training)

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'moving_mean_initializer':
                initializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer':
                initializers.serialize(self.moving_variance_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(BatchNormalizationF16, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

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

        #optimizer = Adam(0.0002, 0.5)
        optimizer = Optimizer()

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
        label = Input(shape=(1,), dtype='float16')
        label_embedding = Flatten()(Embedding(self.num_of_classes, 100)(label))
        label_embedding = K.cast(label_embedding, dtype='float16')

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


